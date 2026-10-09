"""导出完整训练曲线数据到 CSV

- train/val Loss: 取自 runs/run_XXXX 的 tensorboard 记录 (训练时 loss 用的是
  正确概率, 曲线有效)
- val F1/IoU: 训练时记录的 F1/IoU 因多套一层 sigmoid 而无效(恒定),
  这里用每个 epoch 保存的权重在 val 集上重算, 得到真实曲线

用法:
    python export_curves.py                     # 全部 100 个 epoch
    python export_curves.py --end 9             # 只跑前 10 个 (测试)
    中断后重跑会自动跳过 CSV 里已有的 epoch (断点续跑)
"""
import argparse
import csv
import os
import time

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset.dataset1 as dtset
from models.change_classifier import ChangeClassifier as Model
from mymetrics import ChangeDetectionMetrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", default="/home/mvai/Documents/zyn/TinyCD_data")
    ap.add_argument("--rundir", default="runs/run_0002")
    ap.add_argument("--out", default="/home/mvai/Documents/zyn/exp/baseline_v1/curves.csv")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=99, help="最后一个 epoch, 含")
    ap.add_argument("--from-tb", action="store_true",
                    help="直接用 tensorboard 里记录的 val F1/IoU (仅当训练代码的"
                         "指标路径已验证正确时可用, 如 train_ours.py 的 run), 免去逐权重重算")
    args = ap.parse_args()

    # ---- 1. tensorboard 的 loss 曲线 (训练时记录, 有效) ----
    ea = EventAccumulator(args.rundir, size_guidance={"scalars": 0})
    ea.Reload()
    train_loss = {e.step: e.value for e in ea.Scalars("Loss/epoch")}
    val_loss_tb = {e.step: e.value for e in ea.Scalars("Loss_val/epoch")}
    print(f"tensorboard: Loss {len(train_loss)} 点, Loss_val {len(val_loss_tb)} 点")

    if args.from_tb:
        # 训练时指标路径已正确时, F1/IoU 直接取记录值 (与逐权重重算完全等价:
        # 同一权重同一 val 集同一阈值同一累加方式)
        f1_tb = {e.step: e.value for e in ea.Scalars("F1_val class change/epoch")}
        iou_tb = {e.step: e.value for e in ea.Scalars("IoU_val class change/epoch")}
        print(f"tensorboard: F1_val {len(f1_tb)} 点, IoU_val {len(iou_tb)} 点 (--from-tb)")
        all_rows = {
            epc: {"epoch": epc,
                  "train_loss": train_loss.get(epc, ""),
                  "val_loss": val_loss_tb.get(epc, ""),
                  "val_loss_recalc": "",
                  "val_f1": f1_tb.get(epc, ""),
                  "val_iou": iou_tb.get(epc, "")}
            for epc in sorted(set(train_loss) | set(f1_tb))
        }
        if os.path.exists(args.out):
            os.remove(args.out)  # 避免与旧内容混写
        write_csv(args.out, set(), {}, {},
                  [all_rows[e] for e in sorted(all_rows)])
        print(f"完成 -> {args.out}")
        return

    # ---- 2. 准备 val 集与模型 ----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    val_data = dtset.MyDataset(args.datapath, "val")
    loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"val 集: {len(val_data)} 张, device {device}")

    model = Model(bkbn_name="efficientnet_b4", pretrained=False).to(device).eval()
    metric = ChangeDetectionMetrics()

    # ---- 3. 断点续跑: 读已有 CSV ----
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {int(r["epoch"]) for r in csv.DictReader(f)}
        print(f"续跑: CSV 已有 {len(done)} 个 epoch")

    new_rows = []
    for epc in range(args.start, args.end + 1):
        if epc in done:
            continue
        ckpt = os.path.join(args.rundir, f"model_{epc}.pth")
        if not os.path.exists(ckpt):
            print(f"跳过缺失的 {ckpt}")
            continue
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        metric.reset()
        loss_sum = 0.0
        t0 = time.time()
        with torch.no_grad():
            for (ref, test), mask in tqdm(loader, desc=f"epc {epc}", mininterval=10,
                                          leave=False):
                ref = ref.to(device).float()
                test = test.to(device).float()
                mask = mask.to(device).float()
                prob = model(ref, test).squeeze(1)  # 输出即概率, 不再叠加 sigmoid
                loss_sum += torch.nn.functional.binary_cross_entropy(
                    prob.clamp(1e-6, 1 - 1e-6), mask).item()
                metric.update(prob.unsqueeze(1), mask.unsqueeze(1))
        s = metric.compute()
        s["loss"] = loss_sum / len(loader)
        row = {
            "epoch": epc,
            "train_loss": train_loss.get(epc, ""),
            "val_loss": val_loss_tb.get(epc, s["loss"] if not val_loss_tb else ""),
            "val_loss_recalc": s["loss"],
            "val_f1": s["F1"],
            "val_iou": s["IoU"],
        }
        new_rows.append(row)
        print(f"epc {epc:3d} | loss(val,重算) {s['loss']:.4f} | "
              f"F1 {s['F1']:.4f} | IoU {s['IoU']:.4f} | {time.time() - t0:.0f}s", flush=True)

        # 每评完一个 epoch 就落盘, 中断不丢进度
        write_csv(args.out, done | {epc}, train_loss, val_loss_tb, new_rows)

    print(f"\n完成, 共 {len(new_rows)} 个新 epoch -> {args.out}")


def write_csv(path, done_epcs, train_loss, val_loss_tb, new_rows):
    """合并旧数据 + 新数据, 按 epoch 排序后整体重写"""
    all_rows = {r["epoch"]: r for r in new_rows}
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                all_rows[int(r["epoch"])] = r
    fieldnames = ["epoch", "train_loss", "val_loss", "val_loss_recalc",
                  "val_f1", "val_iou"]
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for epc in sorted(all_rows):
            w.writerow(all_rows[epc])
    os.replace(tmp, path)


if __name__ == "__main__":
    main()
