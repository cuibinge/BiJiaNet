import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import torchvision


class TileBuffer(nn.Module):
    def __init__(self, buffer_size=9, feat_dim=256):
        """
        Args:
            buffer_size: 环形缓冲区容量（默认存储8邻域+当前瓦片）
            feat_dim: 输入特征通道数
        """
        super().__init__()
        self.buffer = deque(maxlen=buffer_size)  # 存储格式：(坐标, 特征)

        # 可变形对齐卷积
        self.align_conv = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)
        self.offset_conv = nn.Conv2d(feat_dim * 2, 18, 3, padding=1)  # 生成18通道offset

        # 门控注意力
        self.gate = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 坐标编码器
        self.coord_encoder = nn.Linear(4, feat_dim)  # 编码(x1,y1,x2,y2)坐标差

    def spatial_align(self, src_feat, target_feat):
        """
        特征对齐模块（含可变形卷积）
        Args:
            src_feat: 待对齐的邻域特征 [B,C,H,W]
            target_feat: 目标特征（当前瓦片）[B,C,H,W]
        """
        # 计算坐标差（假设输入为归一化坐标）
        _, _, H, W = target_feat.shape
        coord_grid = self.generate_coord_grid(H, W).to(src_feat.device)

        # 生成offset
        concat_feat = torch.cat([src_feat, target_feat], dim=1)
        offset = self.offset_conv(concat_feat)  # [B,18,H,W]

        # 可变形卷积对齐
        aligned = torchvision.ops.deform_conv2d(
            src_feat,
            offset,
            self.align_conv.weight,
            self.align_conv.bias,
            padding=1
        )
        return aligned

    def generate_coord_grid(self, H, W):
        """
        生成归一化坐标网格
        """
        x_coord = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(1, H, W, -1)
        y_coord = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(1, H, W, -1)
        return torch.cat([x_coord, y_coord], dim=-1).permute(0, 3, 1, 2)  # [1,2,H,W]

    def update_buffer(self, coords, feat):
        """
        更新环形缓冲区
        Args:
            coords: 当前瓦片的网格坐标 (x,y)
            feat: 当前瓦片特征 [B,C,H,W]
        """
        self.buffer.append((coords, feat.detach()))  # 不参与当前梯度计算

    def forward(self, current_feat, current_coords):
        """
        Args:
            current_feat: 当前瓦片特征 [B,C,H,W]
            current_coords: 当前瓦片坐标 (x,y)
        Returns:
            融合后的增强特征 [B,C,H,W]
        """
        B, C, H, W = current_feat.shape
        aligned_feats = []
        coord_embeddings = []

        # 遍历缓冲区寻找相邻瓦片
        for (buf_coords, buf_feat) in self.buffer:
            dx = buf_coords[0] - current_coords[0]
            dy = buf_coords[1] - current_coords[1]

            # 仅处理相邻8邻域
            if abs(dx) <= 1 and abs(dy) <= 1 and (dx, dy) != (0, 0):
                # 特征对齐
                aligned = self.spatial_align(buf_feat, current_feat)
                aligned_feats.append(aligned)

                # 坐标位置编码
                coord_diff = torch.tensor([dx, dy, dx / H, dy / W],
                                          device=current_feat.device)
                coord_emb = self.coord_encoder(coord_diff).view(1, C, 1, 1)
                coord_embeddings.append_emb

                # 无相邻特征时直接返回原特征
                if len(aligned_feats) == 0:
                    return current_feat

        # 融合所有相邻特征
        all_feats = torch.stack(aligned_feats, dim=1)  # [B,N,C,H,W]
        coord_embs = torch.stack(coord_embeddings, dim=1)  # [B,N,C,1,1]

        # 计算注意力权重
        current_expanded = current_feat.unsqueeze(1)  # [B,1,C,H,W]
        energy = torch.cat([current_expanded.expand(-1, len(aligned_feats), -1, -1, -1),
                            all_feats], dim=2)  # [B,N,2C,H,W]
        weights = self.gate(energy.flatten(0, 1)).view(B, -1, 1, H, W)  # [B,N,1,H,W]
        weights = F.softmax(weights, dim=1)

        # 加权融合（含位置编码）
        fused_feat = (all_feats * weights + coord_embs).sum(dim=1)

        # 残差连接
        return current_feat + 0.5 * fused_feat

    def get_buffer_contents(self):
        """调试用：获取缓冲区内容"""
        return [(coords, feat.shape) for (coords, feat) in self.buffer]