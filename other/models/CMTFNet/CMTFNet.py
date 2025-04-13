import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .ResNet import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import ptflops
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class E_FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=5, act_layer=nn.ReLU6, drop=0.):
        super(E_FFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvBNReLU(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.conv1 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=ksize,
                                groups=hidden_features)
        self.conv2 = ConvBNReLU(in_channels=hidden_features, out_channels=hidden_features, kernel_size=3,
                                groups=hidden_features)
        self.fc2 = ConvBN(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = self.fc2(x1 + x2)
        x = self.act(x)

        return x

class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv0_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim //fc_ratio)
        self.bn0_2 = nn.BatchNorm2d(dim // fc_ratio)
        self.conv0_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn0_3 = nn.BatchNorm2d(dim)

        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim // fc_ratio)
        self.bn1_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_3 = nn.BatchNorm2d(dim)

        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn2_2 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_3 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_3 = nn.BatchNorm2d(dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.bn3 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.relu(self.bn0_1(self.conv0_1(x)))
        attn0_2 = self.relu(self.bn0_2(self.conv0_2(attn0_1)))
        attn0_3 = self.relu(self.bn0_3(self.conv0_3(attn0_2)))

        attn1_2 = self.relu(self.bn1_2(self.conv1_2(attn0_1)))
        attn1_3 = self.relu(self.bn1_3(self.conv1_3(attn1_2)))

        attn2_2 = self.relu(self.bn2_2(self.conv2_2(attn0_1)))
        attn2_3 = self.relu(self.bn2_3(self.conv2_3(attn2_2)))

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.relu(self.bn3(self.conv3(attn)))
        attn = attn * u

        pool = self.Avg(attn)

        return pool

class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn

class Block(nn.Module):
    def __init__(self, dim=512, num_heads=16,  mlp_ratio=4, pool_ratio=16, drop=0., dilation=[3, 5, 7],
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Mutilscal_MHSA(dim, num_heads=num_heads, atten_drop=drop, proj_drop=drop, dilation=dilation,
                                   pool_ratio=pool_ratio, fc_ratio=mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim // mlp_ratio)

        self.mlp = E_FFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                         drop=drop)

    def forward(self, x):

        x = x + self.drop_path(self.norm1(self.attn(x)))
        x = x + self.drop_path(self.mlp(x))

        return x

class Fusion(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(Fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = SeparableConvBNReLU(dim, dim, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * x
        x = self.post_conv(x)
        return x

class MAF(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[3, 5, 7], dropout=0., num_classes=6):
        super(MAF, self).__init__()

        self.conv0 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0 = nn.BatchNorm2d(dim//fc_ratio)

        self.conv1_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim//fc_ratio)
        self.bn1_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_2 = nn.BatchNorm2d(dim)

        self.conv2_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim//fc_ratio)
        self.bn2_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_2 = nn.BatchNorm2d(dim)

        self.conv3_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn3_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv3_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn3_2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()

        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )

        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(256, num_classes, kernel_size=1))

    def forward(self, x):
        u = x.clone()

        attn1_0 = self.relu(self.bn0(self.conv0(x)))
        attn1_1 = self.relu(self.bn1_1(self.conv1_1(attn1_0)))
        attn1_1 = self.relu(self.bn1_2(self.conv1_2(attn1_1)))
        attn1_2 = self.relu(self.bn2_1(self.conv2_1(attn1_0)))
        attn1_2 = self.relu(self.bn2_2(self.conv2_2(attn1_2)))
        attn1_3 = self.relu(self.bn3_1(self.conv3_1(attn1_0)))
        attn1_3 = self.relu(self.bn3_2(self.conv3_2(attn1_3)))

        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn

        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn

        attn = attn1_1 + attn1_2 + attn1_3
        attn = self.relu(self.bn4(self.conv4(attn)))
        attn = u * attn

        out = self.head(attn + c_attn + s_attn)

        return out

class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dilation = [[1, 3, 5], [3, 5, 7], [5, 7, 9], [7, 9, 11]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels, 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels, 1)
        self.b4 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[0])

        self.p3 = Fusion(decode_channels)
        self.b3 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[1])

        self.p2 = Fusion(decode_channels)
        self.b2 = Block(dim=decode_channels, num_heads=16, mlp_ratio=4, pool_ratio=16, dilation=dilation[2])

        self.Conv3 = ConvBN(encode_channels[-3], encode_channels[-4], 1)

        self.p1 = Fusion(encode_channels[-4])
        self.seg_head = MAF(encode_channels[-4], fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)

        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w):

        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)

        x = self.b4(res4)

        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.Conv3(x)
        x = self.p1(x, res1)

        x = self.seg_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class DifferentiableSobel(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        # 支持多通道输入（每个通道独立计算Sobel）
        self.sobel_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels)
        # 初始化核（每个通道共享同一个Sobel核）
        sobel_kernel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).float() / 8.0
        sobel_kernel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).float() / 8.0
        self.sobel_x.weight.data = sobel_kernel_x.repeat(in_channels, 1, 1, 1)
        self.sobel_y.weight.data = sobel_kernel_y.repeat(in_channels, 1, 1, 1)
        self.sobel_x.weight.requires_grad_(False)  # 固定核参数
        self.sobel_y.weight.requires_grad_(False)

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient  # 输出形状 [B, C, H, W]


import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """支持批量处理的位置编码模块"""
    def __init__(self, d_model: int):
        super().__init__()
        self.pe = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )

    def forward(self, λ: torch.Tensor) -> torch.Tensor:
        # 输入 λ: [batch, C]
        # 输出: [batch, C, D]
        return self.pe(λ.unsqueeze(-1))  # [batch, C, 1] → [batch, C, D]

class DynamicConvGenerator(nn.Module):
    """支持批量处理的动态卷积核生成器"""

    def __init__(self, num_channels, embed_dim=64, num_heads=3, num_layers=1, patch_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.seq_len = patch_size ** 2 + num_channels + 1

        # 可学习参数（增加批量维度支持）
        self.queries = nn.Parameter(torch.randn(1, self.seq_len, embed_dim))  # [1, S, D]

        # Transformer编码器（支持批量输入）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2 * embed_dim,
            batch_first=True,
            activation=nn.GELU()
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 动态参数生成器
        self.weight_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_channels * patch_size ** 2)
        )
        self.bias_gen = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_channels * embed_dim)
        )

    def forward(self, E_λ_prime: torch.Tensor):
        """
        Args:
            E_λ_prime: [B, C, D] 批量化的波长嵌入
        Returns:
            Kdyn: [B, D, C, P, P] 批量动态卷积核
            Bdyn: [B, C*D] 批量动态偏置
        """
        B, C, D = E_λ_prime.shape
        P = self.patch_size

        # 构建批量查询序列 ====================================
        # 扩展基础查询参数到批量维度 [B, S, D]
        base_queries = self.queries.expand(B, -1, -1)

        # 分割查询参数
        weight_queries = base_queries[:, :P ** 2, :]  # [B, P², D]
        bias_query = base_queries[:, -1:, :]  # [B, 1, D]

        # 拼接波长特征
        queries = torch.cat([
            weight_queries,  # [B, P², D]
            E_λ_prime,  # [B, C, D]
            bias_query  # [B, 1, D]
        ], dim=1)  # [B, P²+C+1, D]

        # Transformer处理 ==================================
        encoded = self.transformer(queries)  # [B, S, D]

        # 生成动态权重 =====================================
        Zw = encoded[:, :P ** 2, :]  # [B, P², D]
        Wdyn = self.weight_gen(Zw)  # [B, P², C*P²]
        Wdyn = Wdyn.view(B, C, P*P, P*P)  # [B, C, P, P]

        # 扩展维度 [B, D, C, P, P]
        Kdyn = Wdyn.unsqueeze(1).expand(-1, D, -1, -1, -1)

        # 生成动态偏置 =====================================
        Zb = encoded[:, -1:, :]  # [B, 1, D]
        Bdyn = self.bias_gen(Zb).view(B, -1)  # [B, C*D]

        return Kdyn, Bdyn

class SpectralAdaptiveConv(nn.Module):
    """支持批量波长输入的光谱自适应卷积"""
    def __init__(self, num_channels, embed_dim=64, patch_size=3):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 编码模块
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.enhancer = nn.Sequential(
            nn.Linear(embed_dim, 2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim)
        )

        # 动态卷积生成器
        self.dyn_conv_gen = DynamicConvGenerator(
            num_channels=num_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        # 深度可分离卷积（支持批量动态核）
        self.dw_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels * embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=patch_size//2,
            groups=num_channels
        )

    def forward(self, I: torch.Tensor, λ: torch.Tensor):
        """
        Args:
            I: [B, C, H, W] 输入图像
            λ: [B, C] 批量波长参数
        Returns:
            Zp: [B, D, H', W'] 输出特征
        """
        B, C, H, W = I.shape
        D = self.embed_dim
        P = self.patch_size

        # 波长编码 ========================================
        E_λ = self.enhancer(self.pos_encoder(λ))  # [B, C, D]

        # 生成动态参数 ====================================
        Kdyn, Bdyn = self.dyn_conv_gen(E_λ)  # [B, D, C, P, P], [B, C*D]

        # 深度卷积特征提取 =================================
        Zp = self.dw_conv(I)  # [B, C*D, H', W']
        Zp = Zp.view(B, C, D, -1)  # [B, C, D, H'W']
        Zp = Zp.permute(0, 2, 1, 3)  # [B, D, C, H'W']

        # 动态偏置注入 ====================================
        Bdyn = Bdyn.view(B, C, D).permute(0, 2, 1)  # [B, D, C]
        Zp = Zp + Bdyn.unsqueeze(-1)  # 广播加法 [B, D, C, H'W']

        # 空间维度恢复 ====================================
        H_out = (H + 2*(P//2) - P) // P + 1  # 准确计算输出尺寸
        W_out = (W + 2*(P//2) - P) // P + 1
        return Zp.mean(dim=2).view(B, D, H_out, W_out)

class CMTFNet(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=512,
                 dropout=0.1,
                 num_classes=6,
                 backbone=ResNet50
                 ):
        super().__init__()

        self.backbone = backbone()
        self.decoder = Decoder(encode_channels, decode_channels, dropout=dropout, num_classes=num_classes)
        self.DifferentiableSobel = DifferentiableSobel(in_channels=2)
        self.SRAM = SpectralAdaptiveConv(num_channels=3, embed_dim=3)
        self.UpSample = nn.Upsample(
                            scale_factor= 2.995,
                            mode='nearest'    # 上采样模式
                        )
    def forward(self, x, λ):
        x = self.SRAM(x, λ)
        x = self.UpSample(x)
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone(x)
        x = self.decoder(res1, res2, res3, res4, h, w)
        # gradient = self.DifferentiableSobel(x)
        return x


if __name__ == '__main__':

    x = torch.randn(8, 3, 512, 512)
    # net = CMTFNet(num_classes=2)
    # flops, params = ptflops.get_model_complexity_info(net, (3, 128, 128), as_strings=True,
    #                                                   print_per_layer_stat=True, verbose=True)
    # print('FLOPs:  ' + flops)
    # print('Params: ' + params)
    num_channels = 3
    start_wavelength = 450  # 起始波长 (nm)
    end_wavelength = 900  # 终止波长 (nm)
    batch_size = 8  # 批量大小

    # 生成单个样本的波长向量
    λ_single = torch.linspace(start_wavelength, end_wavelength, num_channels, dtype=torch.float32)  # [C]

    # 将波长向量扩展到批量维度
    λ_batch = λ_single.unsqueeze(0).expand(batch_size, -1)  # [B, C]

    # 输出：tensor([450.0, 480.0, 510.0, ..., 900.0])
    out = net(x,λ_batch)
    # out,gradient = net(x)
    print(out.shape)
    # print(gradient.shape)
