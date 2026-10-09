"""输出模型复杂度 + 测试集预测黑白图, 结果保存到 exp/baseline_v1

用法:
    python picture.py            # 统计复杂度并保存一张测试集预测示例
    python picture.py --all      # 额外导出整个测试集的预测黑白图
输出:
    exp/baseline_v1/complexity.txt   模型复杂度 (Params / FLOPs)
    exp/baseline_v1/pred_xxx.png     预测黑白图 (0=黑 未变化, 255=白 变化)
    exp/baseline_v1/gt_xxx.png       对应 GT, 便于对照
    exp/baseline_v1/test_preds/      (--all 时) 全部测试集预测 pred_xxx.png
"""
import argparse
import os
import time

import cv2
import numpy as np
import torch
from matplotlib.image import imread
from thop import profile
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset.dataset1 as dtset


def get_model_class(ours: bool):
    if ours:
        from models.change_classifier_ours import ChangeClassifier as Model
    else:
        from models.change_classifier import ChangeClassifier as Model
    return Model

ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true",
                help="导出整个测试集的预测黑白图到 test_preds/")
ap.add_argument("--out-dir", default="/home/mvai/Documents/zyn/exp/baseline_v1",
                help="输出目录 (需含 best.pth)")
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--num-workers", type=int, default=8)
ap.add_argument("--ours", action="store_true",
                help="使用 change_classifier_ours (SGA) 模型")
args = ap.parse_args()

OUT_DIR = args.out_dir
CKPT = os.path.join(OUT_DIR, "best.pth")
DATAPATH = "/home/mvai/Documents/zyn/TinyCD_data"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型 (best.pth)
model = get_model_class(args.ours)(
    bkbn_name="efficientnet_b4", pretrained=False).to(device)
_sd = torch.load(CKPT, map_location="cpu")
if args.ours:  # 过滤 run_0003 早期 checkpoint 的 sga.* 冗余键(旧别名)
    _sd = {k: v for k, v in _sd.items() if not k.startswith("sga.")}
model.load_state_dict(_sd)
model.eval()

# ===== 1. 模型复杂度 (输入 256x256 图像对) =====
dummy_A = torch.randn(1, 3, 256, 256).to(device)
dummy_B = torch.randn(1, 3, 256, 256).to(device)
flops, params = profile(model, inputs=(dummy_A, dummy_B), verbose=False)
params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

report = (
    f"Model: {'TinyCD+SGA (ours)' if args.ours else 'TinyCD (baseline)'}"
    f", backbone=efficientnet_b4\n"
    f"Input: 一对 3x256x256 图像\n"
    f"Params: {params / 1e6:.2f} M (可训练 {params_trainable / 1e6:.2f} M)\n"
    f"FLOPs: {flops / 1e9:.2f} G (thop 统计)\n"
)
print("\n===== 模型复杂度 =====")
print(report)
with open(os.path.join(OUT_DIR, "complexity.txt"), "w") as f:
    f.write(report)

# ===== 2. 测试集预测黑白图 =====
test_data = dtset.MyDataset(DATAPATH, "test")  # test 模式无增强, 结果确定
names = [n.strip("\n") for n in test_data._list_images]


def to_bw(prob):
    """模型输出即概率(末层内置 Sigmoid), 0.5 二值化后放大到 0/255 黑白图"""
    return (prob >= 0.5).cpu().numpy().astype("uint8") * 255


if args.all:
    # 导出整个测试集
    pred_dir = os.path.join(OUT_DIR, "test_preds")
    os.makedirs(pred_dir, exist_ok=True)
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    done, t0 = 0, time.time()
    with torch.no_grad():
        for (ref, test_img), _ in tqdm(loader, desc="export", mininterval=5):
            prob = model(ref.to(device).float(),
                         test_img.to(device).float()).squeeze(1)
            for i, bw in enumerate(to_bw(prob)):
                stem = os.path.splitext(names[done + i])[0]
                cv2.imwrite(os.path.join(pred_dir, f"pred_{stem}.png"), bw)
            done += prob.shape[0]
    print(f"\n共导出 {done} 张预测黑白图 -> {pred_dir} (耗时 {time.time() - t0:.0f}s)")
else:
    # 单张演示: 挑一张有代表性(变化像素占比 1%~50%)的样本, 避免全黑 GT 没意义
    idx = 0
    for i, n in enumerate(names[:200]):
        gt = imread(os.path.join(DATAPATH, "label", n))
        ratio = np.clip(gt * 255, 0, 1).mean()
        if 0.01 <= ratio <= 0.5:
            idx = i
            break

    (ref, test_img), mask = test_data[idx]
    with torch.no_grad():
        prob = model(ref.unsqueeze(0).to(device),
                     test_img.unsqueeze(0).to(device)).squeeze(0).squeeze(0)

    pred_bw = to_bw(prob)
    gt_bw = mask.numpy().astype("uint8") * 255

    stem = os.path.splitext(names[idx])[0]
    pred_path = os.path.join(OUT_DIR, f"pred_{stem}.png")
    gt_path = os.path.join(OUT_DIR, f"gt_{stem}.png")
    cv2.imwrite(pred_path, pred_bw)
    cv2.imwrite(gt_path, gt_bw)

    print("===== 预测示例 =====")
    print(f"样本: {names[idx]} (index {idx}, GT 变化占比 {gt_bw.mean() / 255:.2%})")
    print(f"预测正样本占比: {(prob >= 0.5).float().mean():.2%}")
    print(f"已保存: {pred_path}")
    print(f"已保存: {gt_path}")
