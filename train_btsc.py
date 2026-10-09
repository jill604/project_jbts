import argparse
import os
import shutil

import dataset.dataset1 as dtset
import torch
import numpy as np
import random

from loss_btsc import BiTemporalSymmetryLoss
# 导入你自己的“铁律”评估类
from mymetrics import ChangeDetectionMetrics
from models.change_classifier_ours import ChangeClassifier as Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
        default="/home/mvai/Documents/zyn/TinyCD_data",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        help="log path",
        default="runs",
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use (only applicable to non-distributed training)')

    parsed_arguments = parser.parse_args()

    if not os.path.exists(parsed_arguments.log_path):
        os.makedirs(parsed_arguments.log_path, exist_ok=True)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.log_path)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.log_path = os.path.join(
        parsed_arguments.log_path, "run_%04d" % num_run + "/"
    )

    return parsed_arguments


def train(
    dataset_train,
    dataset_val,
    model,
    criterion,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs=1,
    save_after=1,
    device=torch.device("cpu"),
):

    model = model.to(device)

    # 1. 实例化你自己的铁律评估器（不需要传入任何参数）
    tool4metric = ChangeDetectionMetrics()

    def btsc_step(reference, testimg, mask):
        """BTSC 省显存版训练步 (与 loss_btsc.py 原式梯度完全等价)

        原式: L = 0.5*(BCE(P12,Y) + BCE(P21,Y)) + lambda*MSE(P12,P21)
        恒等分解 (利用 MSE 梯度可按 stop-gradient 拆两侧):
          L = [0.5*BCE12 + lambda*MSE(P12, sg(P21))]     -- 反向后释放图1
            + [0.5*BCE21 + lambda*MSE(P21, sg(P12))]     -- 反向后释放图2
        任一时刻只保留一张计算图 => 显存峰值与单向训练相同 (8G 卡可跑 bs=8)。
        注: P12 的值先以 no_grad 前向取得, 故主干 BN 统计每步更新 3 次
        (比朴素双向前向多 1 次, 等效 EMA 略快, 对结果影响可忽略)。
        """
        with torch.no_grad():  # 先取 P12 的值 (不建图)
            p12_val = model(reference, testimg).squeeze(1).detach()  # [B,H,W]

        p21 = model(testimg, reference).squeeze(1)  # 图2: 逆向 [B,H,W]
        l21 = 0.5 * criterion.bce(p21.clamp(1e-7, 1 - 1e-7), mask) \
            + criterion.lambda_sym * criterion.mse(p21, p12_val)
        l21.backward()

        p12 = model(reference, testimg).squeeze(1)  # 图1: 正向 [B,H,W]
        l12 = 0.5 * criterion.bce(p12.clamp(1e-7, 1 - 1e-7), mask) \
            + criterion.lambda_sym * criterion.mse(p12, p21.detach())
        l12.backward()

        # 返回与原式同量级的监控值 (任务项均值 + 对称项) + 正向预测 (供指标复用)
        with torch.no_grad():
            bce12 = criterion.bce(p12.clamp(1e-7, 1 - 1e-7), mask)
            sym = criterion.mse(p12, p21)
            loss_val = 0.5 * (bce12 + criterion.bce(
                p21.clamp(1e-7, 1 - 1e-7), mask)) + criterion.lambda_sym * sym
        return loss_val, p12.detach()

    def training_phase(epc):
        tool4metric.reset() # 对应你写的 reset 方法
        print(f"Epoch {epc}")
        model.train()
        epoch_loss = 0.0

        # 适配 TinyCD 的 DataLoader 解包：(reference, testimg), mask
        for (reference, testimg), mask in tqdm(
            dataset_train, desc=f"Epoch {epc} [train]"
        ):
            reference = reference.to(device).float()
            testimg = testimg.to(device).float()
            mask = mask.to(device).float()

            optimizer.zero_grad()

            it_loss, preds = btsc_step(reference, testimg, mask)
            optimizer.step()

            epoch_loss += it_loss.item()

            # 指标只用正向输出 P12 (与 v1/v2/ALL 评测协议一致)
            # 模型末层已内置 Sigmoid, 输出即概率, 不能再 sigmoid 一次
            preds = preds.squeeze(1)  # [B,1,H,W] -> [B,H,W]
            tool4metric.update(preds.unsqueeze(1), mask.unsqueeze(1))

        epoch_loss /= len(dataset_train)

        print("Training phase summary")
        print(f"Loss for epoch {epc} is {epoch_loss}")
        writer.add_scalar("Loss/epoch", epoch_loss, epc)

        # 调用你写好的 .compute() 获取指标字典
        scores = tool4metric.compute()
        writer.add_scalar("IoU class change/epoch", scores['IoU'], epc)
        writer.add_scalar("F1 class change/epoch", scores['F1'], epc)

        print(f"IoU class change for epoch {epc} is {scores['IoU']:.4f}")
        print(f"F1 class change for epoch {epc} is {scores['F1']:.4f}\n")
        writer.flush()

        if epc % save_after == 0:
            torch.save(
                model.state_dict(), os.path.join(logpath, f"model_{epc}.pth")
            )

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.reset() # 清空验证集混淆矩阵统计

        with torch.no_grad():
            for (reference, testimg), mask in tqdm(
                dataset_val, desc=f"Epoch {epc} [val]"
            ):
                reference = reference.to(device).float()
                testimg = testimg.to(device).float()
                mask = mask.to(device).float()

                # 验证只走正向 (与历次实验协议一致, val loss 用纯 BCE 可比)
                generated_mask = model(reference, testimg)
                if generated_mask.dim() == 4 and generated_mask.shape[1] == 1:
                    generated_mask = generated_mask.squeeze(1)

                it_loss = torch.nn.functional.binary_cross_entropy(
                    generated_mask.clamp(1e-7, 1 - 1e-7), mask)
                epoch_loss_eval += it_loss.item()

                # 验证阶段统计指标
                preds = generated_mask
                if mask.dim() == 3:
                    tool4metric.update(preds.unsqueeze(1), mask.unsqueeze(1))
                else:
                    tool4metric.update(preds.unsqueeze(1), mask)

        epoch_loss_eval /= len(dataset_val)
        print("Validation phase summary")
        print(f"Loss for epoch {epc} is {epoch_loss_eval}")
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)

        scores = tool4metric.compute()
        writer.add_scalar("IoU_val class change/epoch", scores['IoU'], epc)
        writer.add_scalar("F1_val class change/epoch", scores['F1'], epc)

        print(f"IoU_val class change for epoch {epc} is {scores['IoU']:.4f}")
        print(f"F1_val class change for epoch {epc} is {scores['F1']:.4f}\n")

    for epc in range(epochs):
        training_phase(epc)
        validation_phase(epc)
        scheduler.step()


def run():
    args = parse_arguments()
    writer = SummaryWriter(log_dir=args.log_path)

    trainingdata = dtset.MyDataset(args.datapath, "train")
    validationdata = dtset.MyDataset(args.datapath, "val")
    # num_workers/pin_memory 仅加速数据读取, 不改变训练超参与数学 (bs 仍为 8)
    data_loader_training = DataLoader(trainingdata, batch_size=8, shuffle=True,
                                      drop_last=True, num_workers=8,
                                      pin_memory=True)
    data_loader_val = DataLoader(validationdata, batch_size=8, shuffle=False,
                                 num_workers=8, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    model = Model()
    parameters_tot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of model trainable parameters {parameters_tot}\n")

    criterion = BiTemporalSymmetryLoss(lambda_sym=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00356799066427741,
                                  weight_decay=0.009449677083344786, amsgrad=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100)

    os.makedirs(os.path.join(args.log_path, "models"), exist_ok=True)
    if os.path.exists("./models"):
        try:
            shutil.copytree("./models", os.path.join(args.log_path, "models"), dirs_exist_ok=True)
        except Exception:
            pass

    train(
        data_loader_training,
        data_loader_val,
        model,
        criterion,
        optimizer,
        scheduler,
        args.log_path,
        writer,
        epochs=100,
        save_after=1,
        device=device
    )
    writer.close()


if __name__ == "__main__":
    run()
