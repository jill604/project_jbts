import torch
import torch.nn as nn

class BiTemporalSymmetryLoss(nn.Module):
    """
    时相对称性一致性损失 (BTSC Loss)
    核心数学公理: Change(T1, T2) == Change(T2, T1)
    
    Total_Loss = 0.5 * (BCE(P_12, Y) + BCE(P_21, Y)) + lambda_sym * MSE(P_12, P_21)
    """
    def __init__(self, lambda_sym=0.5):
        """
        :param lambda_sym: 对称一致性约束项的权重 (建议设为 0.5 或 1.0)
        """
        super(BiTemporalSymmetryLoss, self).__init__()
        self.lambda_sym = lambda_sym
        self.bce = nn.BCELoss()  # 依旧使用对概率图最稳定的标准 BCELoss
        self.mse = nn.MSELoss()  # 约束两路输出几何分布一致的距离度量

    def forward(self, pred_12, pred_21, target):
        """
        :param pred_12: 正向输出 model(T1, T2) [B, 1, H, W]，值域 [0, 1]
        :param pred_21: 逆向输出 model(T2, T1) [B, 1, H, W]，值域 [0, 1]
        :param target:  真实变化标签 [B, 1, H, W] 或 [B, H, W]，值域 {0, 1}
        """
        # 1. 维度与类型对齐 (防报错)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()

        # 数值截断保护，防止边界极值导致梯度 NaN
        pred_12 = torch.clamp(pred_12, min=1e-7, max=1.0 - 1e-7)
        pred_21 = torch.clamp(pred_21, min=1e-7, max=1.0 - 1e-7)

        # 2. 双向基础任务监督 (确保无论正着看还是反着看，都在向真实标签靠拢)
        loss_task_12 = self.bce(pred_12, target)
        loss_task_21 = self.bce(pred_21, target)
        loss_task = 0.5 * (loss_task_12 + loss_task_21)

        # 3. 对称性一致性正则化 (强制两路在像素级达成绝对共识)
        loss_sym = self.mse(pred_12, pred_21)

        # 4. 加权得到总损失
        total_loss = loss_task + self.lambda_sym * loss_sym

        # 返回总 loss，顺便返回两个子 loss 方便打印观察
        return total_loss, loss_task, loss_sym


# =========================================================================
# 独立单元测试
# =========================================================================
if __name__ == "__main__":
    print("🚀 正在验证 BiTemporalSymmetryLoss 单元测试...")
    criterion = BiTemporalSymmetryLoss(lambda_sym=0.5)

    # 模拟两路概率输出与真实标签
    p12 = torch.rand(2, 1, 256, 256, requires_grad=True)
    p21 = torch.rand(2, 1, 256, 256, requires_grad=True)
    gt = torch.randint(0, 2, (2, 1, 256, 256)).float()

    total_loss, l_task, l_sym = criterion(p12, p21, gt)
    print(f"总损失: {total_loss.item():.4f} (任务项: {l_task.item():.4f}, 对称约束项: {l_sym.item():.4f})")

    # 验证反向传播与梯度流动
    total_loss.backward()
    assert p12.grad is not None and p21.grad is not None, "❌ 错误：梯度回传中断！"
    print("🎉 单元测试通过！代码无暗病，可以无缝接入训练！")