# metrics.py
import torch

class ChangeDetectionMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        """每个 Epoch 开始前清空统计器"""
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.TN = 0.0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        更新全局混淆矩阵
        :param preds: 模型输出的概率图，形状为 (B, 1, H, W)，取值范围 [0.0, 1.0] （已过 Sigmoid）
        :param labels: 真实标签，形状为 (B, 1, H, W)，取值为 0 或 1
        """
        # 1. 确保在 CPU 进行布尔/数值统计，避免显存泄露
        preds = (preds > 0.5).detach().cpu().int()
        labels = labels.detach().cpu().int()

        # 2. 统计像素个数并累加
        self.TP += ((preds == 1) & (labels == 1)).sum().item()
        self.FP += ((preds == 1) & (labels == 0)).sum().item()
        self.FN += ((preds == 0) & (labels == 1)).sum().item()
        self.TN += ((preds == 0) & (labels == 0)).sum().item()

    def compute(self):
        """
        计算全局微观 (Micro) 指标
        """
        eps = 1e-7  # 防止分母为 0
        
        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.TP / (self.TP + self.FP + self.FN + eps)
        
        return {
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'IoU': float(iou)
        }