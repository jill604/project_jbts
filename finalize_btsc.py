"""BTSC 实验收尾脚本 (训练完成后自动产出全套结果)

产出 (全部进入 Documents/zyn/exp/BTSC/):
  best.pth / last.pth   验证集 F1 最优权重 / 最后一个 epoch 权重
  curves.csv + curves.png  Loss + val F1 全程曲线
  test_preds/           19355 张测试集预测黑白图
  complexity.txt        Params / FLOPs
  metrics.txt           best epoch 与测试集最终指标
  train_BTSC.log + events.tfevents  训练日志

用法 (训练结束后手动跑, 或由守护命令链自动调用):
  python finalize_btsc.py            # 自动定位最新 run 目录
"""
import csv
import glob
import os
import re
import shutil
import subprocess
import sys

PY = sys.executable
ROOT = "/home/mvai/Documents/zyn/TinyCD"
OUT = "/home/mvai/Documents/zyn/exp/BTSC"
SRC_LOG = "/tmp/train_BTSC.log"


def sh(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. 定位本次训练的 run 目录 (最新的)
    rundir = max(
        (d for d in glob.glob(os.path.join(ROOT, "runs", "run_*"))
         if os.path.isdir(d)),
        key=lambda d: int(re.search(r"run_(\d+)", d).group(1)),
    )
    print(f"run 目录: {rundir}")
    assert os.path.exists(os.path.join(rundir, "model_99.pth")), \
        "训练未完成 (缺 model_99.pth), 中止收尾"

    # 2. 曲线数据 (tb 的 F1/val-loss 路径与历次实验一致: 正向 P12 + 纯 BCE, 可比)
    csv_path = os.path.join(OUT, "curves.csv")
    sh(PY, "export_curves.py", "--from-tb", "--rundir", rundir,
       "--out", csv_path)

    # 3. best epoch (val F1 最高) -> best.pth / last.pth
    rows = list(csv.DictReader(open(csv_path)))
    best = max(rows, key=lambda r: float(r["val_f1"]))
    best_epc = int(best["epoch"])
    shutil.copy(os.path.join(rundir, f"model_{best_epc}.pth"),
                os.path.join(OUT, "best.pth"))
    shutil.copy(os.path.join(rundir, "model_99.pth"),
                os.path.join(OUT, "last.pth"))
    print(f"best epoch {best_epc}: val F1 {float(best['val_f1']):.4f}, "
          f"IoU {float(best['val_iou']):.4f}")

    # 4. 曲线图 (val loss 为纯 BCE, 与 v1/v2/ALL 同口径)
    sh(PY, "plot_curves.py", "--csv", csv_path,
       "--out", os.path.join(OUT, "curves.png"),
       "--title", f"TinyCD+SGA+BTSC - training curves "
                  f"({os.path.basename(rundir)}, 100 epochs)",
       "--note", "train loss = BTSC total; val loss/F1 = forward pass + plain "
                 "BCE (comparable across runs). Full data: curves.csv")

    # 5. 测试集全量预测 + 复杂度
    sh(PY, "picture.py", "--all", "--ours", "--out-dir", OUT)

    # 6. 测试集指标 -> metrics.txt (评测 = 正向单向前向, 与历次协议一致)
    r = subprocess.run(
        [PY, "eval_ckpt.py", "--ours", "--ckpt", os.path.join(OUT, "best.pth"),
         "--subset", "test"],
        capture_output=True, text=True, cwd=ROOT)
    with open(os.path.join(OUT, "metrics.txt"), "w") as f:
        f.write(f"run: {rundir}\n"
                f"model: TinyCD + SGA + BiTemporalSymmetryLoss(lambda_sym=0.5)\n"
                f"best epoch: {best_epc} "
                f"(val F1 {float(best['val_f1']):.4f} / "
                f"IoU {float(best['val_iou']):.4f})\n\n{r.stdout}")
    print(r.stdout)

    # 7. 训练日志 (去掉 tqdm 进度条噪音) + tensorboard 事件
    if os.path.exists(SRC_LOG):
        with open(SRC_LOG, errors="ignore") as f, \
                open(os.path.join(OUT, "train_BTSC.log"), "w") as g:
            for line in f.read().replace("\r", "\n").split("\n"):
                if "it/s]" not in line and line.strip():
                    g.write(line + "\n")
    for ev in glob.glob(os.path.join(rundir, "events.out.tfevents.*")):
        shutil.copy(ev, OUT)

    print(f"\n全部完成 -> {OUT}")


if __name__ == "__main__":
    main()
