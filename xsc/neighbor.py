import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FeatureExtractor(nn.Module):
    """共享特征提取器"""

    def __init__(self, in_channels=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, img_tile):
        """输入: [B,C,H,W] 输出: [B,128,H//4,W//4]"""
        return self.encoder(img_tile)


class TileFusionSystem(nn.Module):
    """端到端的瓦片处理系统"""

    def __init__(self, tile_manager):
        super().__init__()
        # 共享特征提取器
        self.extractor = FeatureExtractor()

        # 特征融合模块
        self.fusion = TileFusion(feat_dim=128)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 1)  # 输出分割结果
        )

        # 外部瓦片管理器
        self.tile_manager = tile_manager

    def forward(self, current_tile, current_coord):
        """
        Args:
            current_tile: 当前瓦片图像 [B,C,H,W]
            current_coord: 当前坐标 (x,y)
        """
        # 1. 提取当前瓦片特征
        current_feat = self.extractor(current_tile)  # [B,128,H/4,W/4]

        # 2. 获取相邻瓦片原始图像
        neighbors = self.tile_manager.get_adjacent_tiles(current_coord)

        # 3. 提取相邻瓦片特征
        neighbor_feats = []
        for (dx, dy), neighbor_img in neighbors:
            neighbor_feat = self.extractor(neighbor_img.to(current_tile.device))
            neighbor_feats.append((dx, dy, neighbor_feat))

        # 4. 特征融合
        fused_feat = self.fusion(current_feat, current_coord, neighbor_feats)

        # 5. 解码输出
        output = self.decoder(fused_feat)
        return output


class TileFusion(nn.Module):
    """特征融合模块（修改版）"""

    def __init__(self, feat_dim=128):
        super().__init__()
        # 可变形对齐
        self.align_conv = nn.Conv2d(feat_dim, feat_dim, 3, padding=1)
        self.offset_conv = nn.Conv2d(feat_dim * 2, 18, 3, padding=1)

        # 注意力机制
        self.gate = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 2, 1, 1),
            nn.Sigmoid()
        )

        # 坐标编码器
        self.coord_encoder = nn.Linear(4, feat_dim)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def spatial_align(self, src_feat, target_feat):
        """特征对齐"""
        concat_feat = torch.cat([src_feat, target_feat], dim=1)
        offset = self.offset_conv(concat_feat)
        return torchvision.ops.deform_conv2d(
            src_feat, offset, self.align_conv.weight,
            self.align_conv.bias, padding=1
        )

    def forward(self, current_feat, current_coord, neighbors):
        B, C, H, W = current_feat.shape
        aligned_feats = []
        coord_embeddings = []

        # 处理每个邻居
        for dx, dy, n_feat in neighbors:
            # 特征对齐
            aligned = self.spatial_align(n_feat, current_feat)
            aligned_feats.append(aligned)

            # 坐标编码
            coord_diff = torch.tensor(
                [dx, dy, dx / H, dy / W],
                device=current_feat.device,
                dtype=torch.float
            )
            coord_emb = self.coord_encoder(coord_diff).view(1, C, 1, 1)
            coord_embeddings.append(coord_emb)

        if not aligned_feats:
            return current_feat

        # 特征融合
        all_feats = torch.stack(aligned_feats, dim=1)
        coord_embs = torch.stack(coord_embeddings, dim=1)

        # 注意力计算
        current_expanded = current_feat.unsqueeze(1)
        energy = torch.cat([current_expanded.expand_as(all_feats), all_feats], dim=2)
        weights = F.softmax(self.gate(energy.flatten(0, 1)).view(B, -1, 1, H, W), dim=1)

        # 残差融合
        fused = (all_feats * weights + coord_embs).sum(dim=1)
        return current_feat + self.fusion_weight * fused


class TileManager:
    """瓦片管理器（示例实现）"""

    def __init__(self, tile_dict):
        """
        tile_dict: 字典结构 {(x,y): PIL.Image}
        """
        self.tile_map = tile_dict
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def get_adjacent_tiles(self, center_coord):
        """获取8邻域瓦片"""
        x, y = center_coord
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                coord = (x + dx, y + dy)
                if coord in self.tile_map:
                    img = self.tile_map[coord]
                    tensor = self.transform(img).unsqueeze(0)  # [1,C,H,W]
                    neighbors.append(((dx, dy), tensor))
        return neighbors