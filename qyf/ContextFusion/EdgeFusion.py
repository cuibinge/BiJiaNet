import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 边缘提取分支（保持输入输出通道一致）
        self.edge_extract = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # 特征融合分支（明确输入输出维度）
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),  # 输入in_channels+1
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )

    def forward(self, dec_feat, enc_feat):
        # 在EdgeAwareFusion.forward开始处：
        print(f"Input shapes - dec: {dec_feat.shape}, enc: {enc_feat.shape}")
        assert dec_feat.shape[1] == enc_feat.shape[1], "解码和编码特征通道数必须相同"
        # 维度检查
        assert dec_feat.size(1) == self.in_channels, f"dec_feat应有{self.in_channels}通道，实际得到{dec_feat.size(1)}"
        assert enc_feat.size(1) == self.in_channels, f"enc_feat应有{self.in_channels}通道，实际得到{enc_feat.size(1)}"

        # 边缘提取
        edge_map = self.edge_extract(enc_feat)  # [B,1,H,W]

        # 特征融合
        fused = torch.cat([dec_feat, edge_map], dim=1)  # [B,in_channels+1,H,W]
        return self.feature_fusion(fused)