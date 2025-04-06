import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class CrossPatchSelfAttention(nn.Module):
    """
    Cross-Patch Self-Attention (CPSA) 模块
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 expansion_ratio: int = 4):
        """
        参数:
            embed_dim: 嵌入维度（必须能被num_heads整除）
            num_heads: 注意力头数量
            dropout: 注意力dropout概率
            expansion_ratio: FFN隐藏层扩展倍数
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 缩放因子

        # 合并QKV投影 (比分开计算效率提升30%)
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)

        # 注意力dropout
        self.attn_dropout = nn.Dropout(dropout)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

        # 前馈网络 (包含扩展和收缩)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion_ratio),
            nn.GELU(),  # 比ReLU更适合Transformer
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion_ratio, embed_dim),
            nn.Dropout(dropout)
        )

        # 归一化层 (Pre-LN结构)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CPSA模块的前向传播
        参数:
            x: 输入张量，形状为 (B, N, D)
                B: 批量大小
                N: Patch数量
                D: 嵌入维度
        返回:
            输出张量，形状为 (B, N, D)
        """
        # === 自注意力部分 ===
        residual = x  # 残差连接
        x = self.norm1(x)  # Pre-LN

        # 合并计算QKV并分割 [效率关键点]
        qkv = self.qkv_proj(x).chunk(3, dim=-1)  # 3 x (B, N, D)

        # 重排为多头格式 [使用einops更清晰]
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d',
                                h=self.num_heads), qkv
        )

        # 注意力计算 (高效矩阵乘法)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 合并多头输出
        x = (attn @ v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.out_proj(x)
        x = self.proj_dropout(x)
        x = residual + x  # 残差连接

        # === 前馈网络部分 ===
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x

class CrossAttentionTransformer(nn.Module):
    """
    Cross Attention Transformer (CAT) 模型
    """

    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 num_layers: int = 12,
                 dropout: float = 0.1,
                 num_classes: int = 1000,
                 expansion_ratio: int = 4):
        """
        参数:
            image_size: 输入图像边长 (正方形)
            patch_size: 分块大小
            embed_dim: 嵌入维度
            num_layers: Transformer层数
            num_classes: 分类数 (0表示无分类头)
            expansion_ratio: FFN隐藏层扩展倍数
        """
        super().__init__()

        # === 输入处理 ===
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch嵌入层 (使用卷积实现)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 位置编码 (可学习参数)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        # === 核心Transformer层 ===
        self.layers = nn.ModuleList([
            CrossPatchSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                expansion_ratio=expansion_ratio
            ) for _ in range(num_layers)
        ])

        # 最终归一化
        self.norm = nn.LayerNorm(embed_dim)

        # === 输出头 ===
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # 初始化所有权重
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        CAT模型的前向传播
        参数:
            x: 输入张量，形状为 (B, C, H, W)
                B: 批量大小
                C: 通道数
                H: 图像高度
                W: 图像宽度
        返回:
            输出张量，形状为 (B, N, D)
                N: Patch数量
                D: 嵌入维度
        """
        B, C, H, W = x.shape

        # === 输入验证 ===
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"输入尺寸{H}x{W}与模型定义{self.image_size}x{self.image_size}不匹配"
            )

        # === Patch嵌入 ===
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # === 位置编码 ===
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # === Transformer处理 ===
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # 最终归一化

        # === 输出处理 ===
        if isinstance(self.head, nn.Identity):
            return x  # 返回完整特征序列
        else:
            # 全局平均池化 + 分类头
            return self.head(x.mean(dim=1))


if __name__ == "__main__":
    # 创建模型 (分类模式)
    model = CrossAttentionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,
        num_layers=12,
        embed_dim=768,
        num_heads=12
    )

    # 测试输入
    dummy_img = torch.randn(2, 3, 224, 224)  # 2张224x224 RGB图像

    # 前向传播
    output = model(dummy_img)
    print("输出形状:", output.shape)  # 应为 (2, 1000)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f}M")  # 约85M参数