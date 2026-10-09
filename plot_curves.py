"""绘制训练曲线图 (Loss + Val F1, 双面板避免双轴)

数据来源: curves.csv (由 export_curves.py 生成)
输出:     curves.png
用法:     python plot_curves.py --csv <csv> --out <png> --title <标题>
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="/home/mvai/Documents/zyn/exp/baseline_v1/curves.csv")
ap.add_argument("--out", default="/home/mvai/Documents/zyn/exp/baseline_v1/curves.png")
ap.add_argument("--title", default="TinyCD baseline_v1 - training curves (run_0002, 100 epochs)")
ap.add_argument("--note", default="Note: val F1 recomputed post-hoc; the F1 logged "
                    "during training was invalid (extra sigmoid). Full data: curves.csv")
args = ap.parse_args()
CSV_PATH, OUT_PATH = args.csv, args.out

# 调色板 (dataviz 校验通过: 三槽 all-pairs PASS)
C_TRAIN, C_VAL, C_F1 = "#2a78d6", "#eb6834", "#1baf7a"
C_TEXT, C_TEXT2, C_GRID = "#0b0b0b", "#52514e", "#e5e5e2"
SURFACE = "#fcfcfb"

rows = []
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        rows.append({k: (float(v) if v else None) for k, v in r.items()})
rows.sort(key=lambda r: r["epoch"])
epc = [int(r["epoch"]) for r in rows]
train_loss = [r["train_loss"] for r in rows]
val_loss = [r["val_loss"] for r in rows]
val_f1 = [r["val_f1"] for r in rows]
print(f"{len(rows)} epochs loaded")

plt.rcParams.update({
    "font.size": 9.5,
    "text.color": C_TEXT,
    "axes.edgecolor": C_TEXT2,
    "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT2,
    "ytick.color": C_TEXT2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8.5, 6.2), sharex=True,
    gridspec_kw={"hspace": 0.42},
)

# ---- 面板 1: Loss (train + val) ----
ax1.plot(epc, train_loss, color=C_TRAIN, lw=2, label="Train loss")
ax1.plot(epc, val_loss, color=C_VAL, lw=2, label="Val loss")
ax1.set_title("Loss (BCE)", loc="left", fontsize=11, fontweight="bold")
ax1.set_ylabel("loss")
ax1.legend(frameon=False, loc="upper right", ncols=2)
# 端点直接标注 ( relieving 低对比度风险的直接标注之一 )
ax1.annotate(f"{train_loss[-1]:.3f}", (epc[-1], train_loss[-1]),
             textcoords="offset points", xytext=(-2, 7),
             ha="right", fontsize=8.5, color=C_TEXT2)
ax1.annotate(f"{val_loss[-1]:.3f}", (epc[-1], val_loss[-1]),
             textcoords="offset points", xytext=(-2, -12),
             ha="right", fontsize=8.5, color=C_TEXT2)

# ---- 面板 2: Val F1 (重算) ----
ax2.plot(epc, val_f1, color=C_F1, lw=2)
ax2.set_title("Validation F1 (recomputed from per-epoch checkpoints)",
              loc="left", fontsize=11, fontweight="bold")
ax2.set_ylabel("F1")
ax2.set_xlabel("Epoch")
# 最优点: 标记 + 注释 (>=8px 标记)
bi = max(range(len(epc)), key=lambda i: val_f1[i])
ax2.plot(epc[bi], val_f1[bi], "o", ms=8, mfc=C_F1, mec=SURFACE, mew=2, zorder=5)
ax2.annotate(f"best: epoch {epc[bi]}, F1 {val_f1[bi]:.4f}",
             (epc[bi], val_f1[bi]), textcoords="offset points", xytext=(10, -4),
             fontsize=8.5, color=C_TEXT)
# 端点数值直接标注 (aqua 对比度 WARN 的 relief)
ax2.annotate(f"{val_f1[-1]:.4f}", (epc[-1], val_f1[-1]),
             textcoords="offset points", xytext=(-2, -12),
             ha="right", fontsize=8.5, color=C_TEXT2)

for ax in (ax1, ax2):
    ax.grid(True, color=C_GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle(args.title, fontsize=13, fontweight="bold", x=0.08, ha="left")
fig.text(0.08, 0.005, args.note, fontsize=8, color=C_TEXT2)

fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"saved -> {OUT_PATH}")
