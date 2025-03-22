import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=4, embed_dim=64, patch_size=4):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8):
        super(SwinTransformerBlock, self).__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(2, 0, 1)  # (H*W, B, C)
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        x = x.permute(1, 2, 0).reshape(B, C, H, W)  # (B, C, H, W)
        return x

class CustomSwinTransformer(nn.Module):
    def __init__(self, input_channels=4, output_channels=32, img_size=128, embed_dim=64, patch_size=4, num_heads=4, num_swin_blocks=4):
        super(CustomSwinTransformer, self).__init__()
        
        # Initial patch embedding
        self.patch_embed = PatchEmbedding(in_channels=input_channels, embed_dim=embed_dim, patch_size=patch_size)
        
        # Stack of Swin Transformer blocks
        self.swin_blocks = nn.ModuleList([SwinTransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(num_swin_blocks)])
        
        # Second embedding layer to learn more complex representations
        self.second_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        
        # Convolution to match the output channels
        self.conv = nn.Conv2d(embed_dim, output_channels, kernel_size=1, stride=1, padding=0)
        
        # Normalization Layer
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Expecting x to have shape (B, 4, 128, 128)
        x = self.patch_embed(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        
        # Apply multiple Swin Transformer blocks with residual connections
        for swin_block in self.swin_blocks:
            x = swin_block(x) + x  # Residual connection to help with gradient flow and generalization
        
        # Apply second embedding layer
        x = self.second_embed(x)
        
        # Normalize the features
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)  # (B, C, H, W)
        
        # Upsample back to original size
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        
        # Adjust the output channels to desired output
        x = self.conv(x)  # (B, output_channels, 128, 128)
        
        return x

# Testing the model
if __name__ == "__main__":
    model = CustomSwinTransformer()
    input_tensor = torch.randn(1, 4, 128, 128)  # Batch size B=1, Channels=4, Height=128, Width=128
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output: (1, 32, 128, 128)
