from typing import List
import torchvision
from models.layers import MixingMaskAttentionBlock, PixelwiseLinear, UpMask
from torch import Tensor
from torch.nn import Module, ModuleList, Sigmoid
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


class ChangeClassifier(Module):
    def __init__(
        self,
        bkbn_name="efficientnet_b4",
        pretrained=True,
        output_layer_bkbn="3",
        freeze_backbone=False,
    ):
        super().__init__()
        
        # Load the pretrained backbone according to parameters:
        self._backbone = _get_backbone(
            bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
        )

        # Initialize mixing blocks:
        self._first_mix = MixingMaskAttentionBlock(6, 3, [3, 10, 5], [10, 5, 1])
        self._mixing_mask = ModuleList(
            [
                MixingMaskAttentionBlock(48, 24, [24, 12, 6], [12, 6, 1]),
                MixingMaskAttentionBlock(64, 32, [32, 16, 8], [16, 8, 1]),
                # 最深层用 SGA 替换原 MixingBlock(112, 56): 二者接口一致
                # ((F1, F2) -> 融合特征), 输出同为 56 通道。双时相特征对只在
                # encoder 的 mixing 阶段存在(decoder 之后已是单张量, SGA 的
                # |F1-F2| 差异门控会退化失效), 故插入在此处
                SGAModule(56),
            ]
        )

        # Initialize Upsampling blocks:
        self._up = ModuleList(
            [
                UpMask(2, 56, 64),
                UpMask(2, 64, 64),
                UpMask(2, 64, 32),
            ]
        )
        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features = self._encode(ref, test)
        latents = self._decode(features)
        return self._classify(latents)

    def _encode(self, ref, test) -> List[Tensor]:
        features = [self._first_mix(ref, test)]
        for num, layer in enumerate(self._backbone):
            ref, test = layer(ref), layer(test)
            if num != 0:
                features.append(self._mixing_mask[num - 1](ref, test))
        return features

    def _decode(self, features) -> Tensor:
        upping = features[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping = self._up[i](upping, features[j])
        return upping


def _get_backbone(
    bkbn_name, pretrained, output_layer_bkbn, freeze_backbone
) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features

    # Slicing it:
    derived_model = ModuleList([])
    for name, layer in entire_model.named_children():
        derived_model.append(layer)
        if name == output_layer_bkbn:
            break

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


if __name__ == "__main__":
    print("🚀 测试魔改后的 TinyCD_Ours 是否能正常工作...")
    model = ChangeClassifier(pretrained=False) # 冒烟测试无需下载预训练
    model.eval()
    
    # 模拟双时相输入
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    out = model(x1, x2)
    print(f"输入形状: {x1.shape}")
    print(f"输出形状: {out.shape}")
    print(f"输出值域: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    assert out.shape == (2, 1, 256, 256), "❌ 输出尺寸不匹配！"
    print("🎉 恭喜！TinyCD_Ours 手术大获成功，完全没有维度错误！")