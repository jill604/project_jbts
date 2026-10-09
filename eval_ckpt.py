"""快速评测已有 checkpoint, 得到有效的 F1/IoU (无需重新训练)

背景: run_0002 训练时指标计算多套了一层 sigmoid, 导致记录的 F1/IoU
恒定无效(全图判为变化); 但损失是直接用模型概率算的(量级可证: 双重
sigmoid 下 loss 不可能低于 ~0.62, 实际收到 0.087), 权重本身没问题。
本脚本按正确方式(模型输出即概率, 不再叠加 sigmoid)重算指标。

用法:
    python eval_ckpt.py --epochs 20,40,60,80,99 --subset val  # 扫多个 epoch 取最优
    python eval_ckpt.py --epochs 99 --subset test             # 用指定 epoch 跑测试集
    python eval_ckpt.py --epochs 99 --subset val --limit 500  # 只评前 500 张快速估计
"""
import argparse
import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import dataset.dataset1 as dtset
from mymetrics import ChangeDetectionMetrics


def get_model_class(ours: bool):
    """baseline 或 ours(SGA) 版模型; ours 版含 sga 语义别名, 与 run_0003 起的
    checkpoint 的重复键 (sga.* 与 _mixing_mask.2.*) 完全匹配"""
    if ours:
        from models.change_classifier_ours import ChangeClassifier as Model
    else:
        from models.change_classifier import ChangeClassifier as Model
    return Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", default="/home/mvai/Documents/zyn/TinyCD_data")
    ap.add_argument("--ckpt", default=None,
                    help="直接指定 checkpoint 路径, 如 exp/baseline_v1/best.pth (优先于 --rundir/--epochs)")
    ap.add_argument("--rundir", default="runs/run_0002")
    ap.add_argument("--epochs", default="20,40,60,80,99",
                    help="要评测的 epoch 编号, 逗号分隔")
    ap.add_argument("--subset", default="val", choices=["train", "val", "test"])
    ap.add_argument("--ours", action="store_true",
                    help="评测 change_classifier_ours (SGA) 模型, 如 runs/run_0003")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只评前 N 个样本")
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = dtset.MyDataset(args.datapath, args.subset)
    n_total = len(data)
    if args.limit > 0:
        data = Subset(data, range(min(args.limit, n_total)))
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    print(f"子集: {args.subset} ({len(data)}/{n_total} 张)\n")

    # pretrained=False: 权重马上会被 checkpoint 覆盖, 不必加载预训练 backbone
    model = get_model_class(args.ours)(
        bkbn_name="efficientnet_b4", pretrained=False).to(device).eval()
    metric = ChangeDetectionMetrics()

    results = []
    if args.ckpt:
        ckpt_list = [(args.ckpt, args.ckpt)]
    else:
        ckpt_list = [(str(epc), f"{args.rundir}/model_{epc}.pth")
                     for epc in [int(e) for e in args.epochs.split(",")]]
    for label, ckpt_path in ckpt_list:
        sd = torch.load(ckpt_path, map_location="cpu")
        if args.ours:  # run_0003 早期 checkpoint 含 sga.* 冗余键(旧别名), 过滤后加载
            sd = {k: v for k, v in sd.items() if not k.startswith("sga.")}
        model.load_state_dict(sd)
        metric.reset()

        loss_sum, first_batch = 0.0, True
        t0 = time.time()
        with torch.no_grad():
            for (ref, test), mask in tqdm(loader, desc=f"{label} [{args.subset}]"):
                ref = ref.to(device).float()
                test = test.to(device).float()
                mask = mask.to(device).float()

                prob = model(ref, test).squeeze(1)  # 输出即概率(末层内置 Sigmoid), 不再叠加
                if first_batch:  # 首批打印分布, 顺便确认无坍塌
                    print(f"  概率分布: min={prob.min():.4f} max={prob.max():.4f} "
                          f"mean={prob.mean():.4f} 正样本比例={(prob >= 0.5).float().mean():.4f}")
                    first_batch = False
                loss_sum += torch.nn.functional.binary_cross_entropy(
                    prob.clamp(1e-6, 1 - 1e-6), mask).item()
                metric.update(prob.unsqueeze(1), mask.unsqueeze(1))

        s = metric.compute()
        s["loss"] = loss_sum / len(loader)
        results.append((label, s))
        print(f"{label} | loss {s['loss']:.4f} | P {s['Precision']:.4f} | "
              f"R {s['Recall']:.4f} | F1 {s['F1']:.4f} | IoU {s['IoU']:.4f} | "
              f"{time.time() - t0:.0f}s\n")

    best = max(results, key=lambda r: r[1]["F1"])
    print(f"最优: epoch {best[0]}  F1={best[1]['F1']:.4f}  IoU={best[1]['IoU']:.4f}  "
          f"(subset={args.subset})")


if __name__ == "__main__":
    main()
