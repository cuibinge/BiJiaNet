import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGRU(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        """
        Args:
            in_channels: 输入特征图的通道数
            hidden_dim: 隐藏状态的维度（通常小于输入通道）
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 空间特征转换
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU()
        )

        # GRU核心计算单元
        self.gru_cell = nn.GRUCell(
            input_size=hidden_dim * 3,  # 当前特征+坐标编码
            hidden_size=hidden_dim
        )

        # 特征重建
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1),
            nn.Dropout2d(0.1)
        )

        # 坐标编码器
        self.coord_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

    def _get_coord_emb(self, feat):
        """生成坐标编码特征"""
        B, C, H, W = feat.shape
        # 生成归一化网格坐标
        x_coord = torch.linspace(-1, 1, W, device=feat.device).view(1, 1, W).expand(B, H, W)
        y_coord = torch.linspace(-1, 1, H, device=feat.device).view(1, H, 1).expand(B, H, W)
        coord = torch.stack([x_coord, y_coord, x_coord * H, y_coord * W], dim=-1)  # [B,H,W,4]
        return self.coord_encoder(coord).permute(0, 3, 1, 2)  # [B,C,H,W]

    def forward(self, current_feat, prev_hidden=None):
        """
        Args:
            current_feat: 当前时刻特征 [B,C,H,W]
            prev_hidden: 前一时刻隐藏状态 [B,D,H,W]
        Returns:
            output_feat: 增强后的特征 [B,C,H,W]
            new_hidden: 更新后的隐藏状态 [B,D,H,W]
        """
        B, C, H, W = current_feat.shape

        # 初始化隐藏状态
        if prev_hidden is None:
            prev_hidden = torch.zeros(B, self.hidden_dim, H, W,
                                      device=current_feat.device)

        # 空间特征投影
        spatial_feat = self.spatial_proj(current_feat)  # [B,D,H,W]

        # 坐标编码
        coord_emb = self._get_coord_emb(current_feat)  # [B,D,H,W]

        # 拼接输入特征
        gru_input = torch.cat([
            spatial_feat,
            coord_emb,
            prev_hidden
        ], dim=1)  # [B,3D,H,W]

        # 逐位置GRU计算
        new_hidden = []
        for h in range(H):
            for w in range(W):
                # 提取当前位置特征 [B,3D]
                loc_feat = gru_input[:, :, h, w]
                # 对应位置历史状态 [B,D]
                loc_prev = prev_hidden[:, :, h, w]
                # GRU计算
                loc_hidden = self.gru_cell(loc_feat, loc_prev)
                new_hidden.append(loc_hidden)

        # 重组空间维度 [B,H,W,D] -> [B,D,H,W]
        new_hidden = torch.stack(new_hidden, dim=1).view(B, H, W, self.hidden_dim)
        new_hidden = new_hidden.permute(0, 3, 1, 2).contiguous()

        # 特征重建
        output_feat = self.output_proj(new_hidden) + current_feat  # 残差连接

        return output_feat, new_hidden

    def reset_hidden(self, batch_size, spatial_size, device='cpu'):
        """初始化隐藏状态（用于序列开始）"""
        H, W = spatial_size
        return torch.zeros(batch_size, self.hidden_dim, H, W, device=device)