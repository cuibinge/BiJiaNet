from typing import Dict, List, Tuple, Union
import subprocess, sys
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

# =====================  Neighbour fusion blocks  ===========================
class FeatureFusion(nn.Module):
    """Concatenate current + 4 neighbours, add direction embeddings, 1×1 conv."""

    def __init__(self, feat_ch: int = 32, max_neighbors: int = 4, positional: bool = True):
        super().__init__()
        self.max_n = max_neighbors
        self.positional = positional

        if positional:
            self.dir_embed = nn.Parameter(torch.zeros(max_neighbors, feat_ch, 1, 1))
            nn.init.xavier_uniform_(self.dir_embed)
        else:
            self.register_parameter("dir_embed", None)

        self.conv1x1 = nn.Conv2d(feat_ch * (1 + max_neighbors), feat_ch, 1)

    def forward(self, cur: torch.Tensor, neigh: List[torch.Tensor]):
        # pad
        neigh_pad = neigh + [torch.zeros_like(cur) for _ in range(self.max_n - len(neigh))]
        if self.positional:
            neigh_pad = [neigh_pad[i] + self.dir_embed[i] for i in range(self.max_n)]
        stacked = torch.cat([cur] + neigh_pad, 1)
        return self.conv1x1(stacked)

class EnhancedFeatureFusion(nn.Module):
    """
    5×32 → DW-3×3 → PW-1×1 → SE → Direction-gated加权 → 32
    """
    def __init__(self, feat_ch=32, max_neighbors=4):
        super().__init__()
        self.max_n = max_neighbors          # 4
        self.dir_embed = nn.Parameter(torch.zeros(max_neighbors, feat_ch, 1, 1))
        nn.init.xavier_uniform_(self.dir_embed)

        # depthwise 3×3
        self.dw = nn.Conv2d(feat_ch * (1+max_neighbors),
                            feat_ch * (1+max_neighbors),
                            3, padding=1, groups=feat_ch*(1+max_neighbors))
        # point-wise 1×1
        self.pw = nn.Conv2d(feat_ch * (1+max_neighbors), feat_ch, 1)

        # Squeeze-and-Excitation on concat tensor
        self.se_fc1 = nn.Conv2d(feat_ch*(1+max_neighbors), feat_ch//4, 1)
        self.se_fc2 = nn.Conv2d(feat_ch//4, feat_ch*(1+max_neighbors), 1)

        # direction-specific gate (learn 1 scalar per dir)
        self.dir_w = nn.Parameter(torch.zeros(max_neighbors))  # α_i = σ(w_i·gap(fi))

    def forward(self, cur, neighs):
        neigh_pad = neighs + [torch.zeros_like(cur) for _ in range(self.max_n - len(neighs))]
        neigh_pad = [neigh_pad[i] + self.dir_embed[i] for i in range(self.max_n)]
        concat = torch.cat([cur] + neigh_pad, 1)  # (B,160,H,W)

        # depthwise + pointwise
        x = self.pw(self.dw(concat))  # (B,32,H,W)

        # -------- SE ----------
        se = F.adaptive_avg_pool2d(concat, 1)
        se = torch.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        x = x * se[:, :32]  # (B,32,H,W)

        # -------- edge hint ----------
        # 取邻居平均，得到边缘概率图 (B,1,H,W)
        edge_hint = torch.stack([n.mean(1, keepdim=True) for n in neigh_pad], 1).max(1)[0]
        x = x * (1 + 0.5 * torch.sigmoid(edge_hint))  # 只放大边缘

        # -------- direction gate ----------
        gaps = torch.stack([F.adaptive_avg_pool2d(n, 1).squeeze(-1).squeeze(-1)
                            for n in neigh_pad], dim=1)  # (B,4,32)
        alpha = torch.sigmoid(self.dir_w).view(1, self.max_n, 1)
        gate = (gaps.mean(-1, keepdim=True) * alpha).sum(1)  # (B,1)
        x = x * (1 + gate.unsqueeze(-1).unsqueeze(-1))

        return x


# ---------------------------------------------------------------------------
# ❷  Trainable wrapper – FusionNestedUNet
# ---------------------------------------------------------------------------
class FusionNestedUNet(nn.Module):
    """NestedUNet backbone + 4‑direction feature fusion → logits."""

    def __init__(self, num_classes: int = 1, in_channels: int = 3, deep_supervision: bool = False):
        super().__init__()
        self.backbone = NestedUNet(num_classes=num_classes, in_ch=in_channels, deep_supervision=deep_supervision)
        self.fusion   = EnhancedFeatureFusion(feat_ch=32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)
        self.adapt = nn.Conv2d(num_classes, 32, 1)  # 新加

    def _encode(self, x):
        out = self.backbone(x)
        feat = out[-1] if isinstance(out,(list,tuple)) else out     # (B,C,H,W)

        # ↓ 若 C≠32 做通道映射（保持）
        if feat.shape[1] != 32:
            feat = self.adapt(feat)

        return feat                                                 # 32, H, W

    def forward(self, x):
        if x.dim()!=5 or x.size(1)<5:
            raise ValueError("Expect (B,5,C,H,W)")

        center = x[:,0]                                             # (B,C,H,W)
        neighs = [x[:,i] for i in range(1,5)]

        feat_c   = self._encode(center)                             # (B,32,H,W)

        # ── 新增：邻居降到 1/2 分辨率再上采 ───────────────
        neigh_feats = []
        for n in neighs:
            f = self._encode(n)                     # (B,32,H,W)
            f = F.avg_pool2d(f, 2)                  # (B,32,H/2,W/2)
            f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=False)
            neigh_feats.append(f)                   # 仍然 (B,32,H,W)
        # ──────────────────────────────────────────

        fused  = self.fusion(feat_c, neigh_feats)
        logits = self.seg_head(fused)
        return logits
