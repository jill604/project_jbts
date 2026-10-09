"""数据管道 & 模型预测分布自检脚本 (独立可运行)

用法:
    python check_debug.py                          # 默认加载 runs 下最新 checkpoint
    python check_debug.py --modelpath runs/run_0002/model_99.pth
    python check_debug.py --modelpath none         # 只检查数据管道(随机初始化模型)
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

import dataset.dataset1 as dtset
from models.change_classifier import ChangeClassifier as Model


def find_latest_checkpoint(root="runs"):
    """在 runs/run_XXXX 下找 epoch 最大的 model_XX.pth"""
    if not os.path.isdir(root):
        return None
    best = None
    for run in sorted(os.listdir(root)):
        run_dir = os.path.join(root, run)
        if not os.path.isdir(run_dir):
            continue
        for f in os.listdir(run_dir):
            if f.startswith("model_") and f.endswith(".pth"):
                epc = int(f[len("model_"):-len(".pth")])
                if best is None or epc > best[0]:
                    best = (epc, os.path.join(run_dir, f))
    return best[1] if best else None


parser = argparse.ArgumentParser()
parser.add_argument("--datapath", type=str,
                    default="/home/mvai/Documents/zyn/TinyCD_data")
parser.add_argument("--modelpath", type=str, default="auto",
                    help="'auto' = runs 下最新 checkpoint, 'none' = 不加载权重")
parser.add_argument("--batch-size", type=int, default=8)
args = parser.parse_args()

# ===== 0. 构建数据管道 / 模型 / 设备 (与 train.py 保持一致) =====
train_data = dtset.MyDataset(args.datapath, "train")
train_loader = DataLoader(train_data, batch_size=args.batch_size,
                          shuffle=True, drop_last=True)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Device: {device}, 训练集样本数: {len(train_data)}")

model = Model()
if args.modelpath == "auto":
    ckpt = find_latest_checkpoint()
    if ckpt:
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"已加载 checkpoint: {ckpt}\n")
    else:
        print("⚠️ 未找到任何 checkpoint,使用随机初始化模型\n")
elif args.modelpath != "none":
    model.load_state_dict(torch.load(args.modelpath, map_location="cpu"))
    print(f"已加载 checkpoint: {args.modelpath}\n")
model.to(device)

# 1. 从你的 train_loader 中抽一个 Batch
batch = next(iter(train_loader))
img_A = batch[0][0]   # (reference, testimg), mask 中的 reference
img_B = batch[0][1]   # testimg
target = batch[1]     # mask

print("=== [1] 数据管道检查 ===")
print(f"Image A 形状: {img_A.shape}, 值域: [{img_A.min():.2f}, {img_A.max():.2f}]")
print(f"Target  形状: {target.shape}, 唯一值集合: {torch.unique(target).tolist()}")

# 检查 Label 是否合格 (必须只有 0 和 1)
assert set(torch.unique(target).tolist()).issubset({0.0, 1.0}), "❌ 致命错误：Label 包含 0 和 1 之外的值（可能是255）！"

# 计算这组数据中变化像素的占比
pos_ratio = (target > 0).float().mean().item() * 100
print(f"当前 Batch 中变化像素占比: {pos_ratio:.2f}% (如果小于 1%，说明极度不平衡)")

print("\n=== [2] 模型预测分布检查 ===")
# 注意: TinyCD 末层已内置 Sigmoid (change_classifier.py:43), 模型输出本身就是概率,
# 不能再套一层 torch.sigmoid() —— 否则概率被压回 [0.5, sigmoid(1)]，产生"坍塌"假象
model.eval()
with torch.no_grad():
    out = model(img_A.to(device), img_B.to(device))
    if isinstance(out, (tuple, list)): out = out[0]
    probs = out  # 已是概率

print(f"模型输出(概率) 最小值: {probs.min().item():.4f}")
print(f"模型输出(概率) 最大值: {probs.max().item():.4f}")
print(f"模型输出(概率) 平均值: {probs.mean().item():.4f}")
print(f"模型预测为正样本(>0.5)的像素比例: {(probs >= 0.5).float().mean().item() * 100:.2f}%")
print(f"(对照) 真实变化像素比例: {pos_ratio:.2f}%")

pos_pred = (probs >= 0.5).float().mean().item() * 100
if probs.max().item() < 0.5:
    print("❌ 确诊：模型彻底坍塌！它对所有像素的预测概率都在 0.5 以下，全图都在预测 0（全黑）！")
elif pos_pred > 99.9:
    print("❌ 确诊：模型反向坍塌！它对所有像素的预测概率都在 0.5 以上，全图都在预测 1（全白）！")
elif probs.max().item() - probs.min().item() < 0.1:
    print("⚠️ 警告：预测概率几乎无区分度（max-min < 0.1），模型输出接近常数，疑似坍塌！")
else:
    print("✅ 模型输出分布正常：概率范围覆盖 [0,1]，有区分度，未见坍塌")