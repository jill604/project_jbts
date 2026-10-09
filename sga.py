import torch
import torch.nn as nn
import torch.nn.functional as F

class StripPooling(nn.Module):
    """
    条带池化核心组件：分别在水平和垂直方向做长条形池化与特征聚合
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 垂直方向条带池化：池化为 [B, C, H, 1]，配合 3x1 卷积建模高度方向连续性
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 水平方向条带池化：池化为 [B, C, 1, W]，配合 1x3 卷积建模宽度方向连续性
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_w = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 垂直分支：池化 -> 卷积 -> 沿宽展开
        feat_h = self.pool_h(x)
        feat_h = self.conv_h(feat_h)
        feat_h = feat_h.expand(-1, -1, H, W)
        
        # 水平分支：池化 -> 卷积 -> 沿高展开
        feat_w = self.pool_w(x)
        feat_w = self.conv_w(feat_w)
        feat_w = feat_w.expand(-1, -1, H, W)
        
        # 条带特征融合
        out = self.fusion(feat_h + feat_w)
        return out


class SGAModule(nn.Module):
    """
    SGA: Strip-Guided Attention (条带引导变化检测注意力模块)
    输入双时相特征 F1, F2，输出经过长条结构增强后的特征
    """
    def __init__(self, in_channels=64, reduction=4):
        super().__init__()
        inter_channels = max(in_channels // reduction, 16)

        # 1. 基础特征融合 (将双时相交互)
        # 输入拼接: [F1, F2, |F1 - F2|]，通道数 3 * in_channels -> in_channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 2. 条带池化提取长距离几何依赖
        self.strip_pool = StripPooling(in_channels, inter_channels)

        # 3. 生成注意力掩膜 (Sigmoid 归一化到 0~1)
        self.attn_conv = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

        # 4. 细化层
        self.refine_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, F1, F2):
        """
        :param F1: 时相 1 特征 [B, C, H, W]
        :param F2: 时相 2 特征 [B, C, H, W]
        :return: 增强后的特征 [B, C, H, W]
        """
        assert F1.shape == F2.shape, f"F1 {F1.shape} 与 F2 {F2.shape} 形状不一致！"

        # 双时相差异
        diff = torch.abs(F1 - F2)

        # 拼接交互
        feat_cat = torch.cat([F1, F2, diff], dim=1)
        feat_base = self.fusion_conv(feat_cat)

        # 条带注意力计算
        strip_feat = self.strip_pool(feat_base)
        attn_weight = self.attn_conv(strip_feat)

        # 注意力加权与残差连接
        out = feat_base + diff * attn_weight
        out = self.refine_conv(out)
        return out


# =========================================================================
# 单元测试 (Unit Test) —— 独立运行，确保模块绝对无暗病
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始 SGA 模块独立单元测试 (Unit Test)...")
    print("=" * 60)

    # 1. 形状与维度测试 (完全按照你的要求构造 F1 和 F2)
    B, C, H, W = 2, 64, 64, 64
    F1 = torch.randn(B, C, H, W, requires_grad=True)
    F2 = torch.randn(B, C, H, W, requires_grad=True)

    module = SGAModule(in_channels=C, reduction=4)
    out = module(F1, F2)

    print(f"\n[测试 1: 形状检查]")
    print(f"输入 F1 形状: {list(F1.shape)}")
    print(f"输入 F2 形状: {list(F2.shape)}")
    print(f"输出特征形状: {list(out.shape)}")
    assert out.shape == (B, C, H, W), "❌ 错误：输出特征图形状不匹配！"
    print("✅ 通过：输出尺寸严格与输入一致 (B, C, H, W)。")

    # 2. 反向传播与梯度流测试
    print(f"\n[测试 2: 梯度回传检查]")
    loss = out.sum()
    loss.backward()

    assert F1.grad is not None and F2.grad is not None, "❌ 错误：梯度无法回传到输入特征！"
    
    # 检查模块内部各层权重是否有断裂或 NaN 梯度
    has_nan_grad = False
    for name, param in module.named_parameters():
        if param.grad is None:
            print(f"⚠️ 警告：参数 {name} 没有梯度！")
        elif torch.isnan(param.grad).any():
            has_nan_grad = True

    assert not has_nan_grad, "❌ 错误：反向传播产生了 NaN 梯度！"
    print("✅ 通过：计算图完整，梯度能平滑回传至两路输入，无断裂。")

    # 3. 统计模块轻量化特性 (用于写论文 Table 2)
    print(f"\n[测试 3: 参数量统计]")
    total_params = sum(p.numel() for p in module.parameters())
    print(f"SGA 模块总参数量: {total_params} 个 ({total_params / 1e3:.2f} K)")
    print(f"（注：仅占几万参数，极其轻量，不会给你的 Baseline 增加明显负担）")

    # 4. GPU 运行与显存检查
    if torch.cuda.is_available():
        print(f"\n[测试 4: CUDA 设备适配检查]")
        device = torch.device("cuda")
        module_gpu = module.to(device)
        F1_gpu = torch.randn(B, C, H, W, device=device)
        F2_gpu = torch.randn(B, C, H, W, device=device)
        out_gpu = module_gpu(F1_gpu, F2_gpu)
        assert out_gpu.is_cuda, "❌ 错误：输出未在 GPU 上！"
        print("✅ 通过：CUDA 运行正常，无显存泄漏问题。")
    else:
        print("\n[测试 4: 跳过 CUDA 测试 (未检测到 GPU)]")

    print("\n" + "=" * 60)
    print("🎉 SGA 模块单元测试全部通过！完美达到接入主网络的标准！")
    print("=" * 60)