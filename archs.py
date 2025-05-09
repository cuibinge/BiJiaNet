from torch import nn
import torch.nn.functional as F
from ContextFusion.BUF import TileBuffer
from ContextFusion.EdgeFusion import EdgeAwareFusion
from ContextFusion.FeatureGRU import FeatureGRU
__all__ = ['UNet', 'NestedUNet', 'SFFNet', 'DynamicNestedUNet']

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 动态调整隐藏层维度
        hidden_dim = max(d_model * 2, 4)  # 示例：使用扩展维度

        self.pe = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model)
        )

        # 初始化权重
        for layer in self.pe:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, λ):
        # 输入λ形状：[batch, channels] -> [batch, channels, 1]
        return self.pe(λ.unsqueeze(-1))

class DynamicConvGenerator(nn.Module):
    """支持批量处理的动态卷积核生成器"""

    def __init__(self, num_channels, embed_dim=64, num_heads=4, num_layers=1, patch_size=3):
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

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class DynamicVGGBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = VGGBlock(in_c, out_c, out_c)
        self.buffer_fusion = TileBuffer(feat_dim=out_c)  # 环形缓冲区融合

    def forward(self, x, neighbor_feats=None):
        x = self.conv(x)
        if neighbor_feats is not None:
            x = self.buffer_fusion(x, neighbor_feats)
        return x


class EdgeEnhancedConcat(nn.Module):
    def __init__(self, enc_channels, dec_channels):
        super().__init__()
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels

        # 通道调整卷积（将解码特征通道数匹配编码特征）
        self.channel_adjust = nn.Conv2d(dec_channels, enc_channels, kernel_size=1)

        # 边缘融合模块（输入为编码特征通道数）
        self.edge_fusion = EdgeAwareFusion(enc_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, enc, dec):
        # 1. 上采样解码特征
        dec_up = self.up(dec)

        # 2. 调整通道数匹配编码特征
        dec_adjusted = self.channel_adjust(dec_up)

        # 3. 边缘增强融合
        dec_enhanced = self.edge_fusion(dec_adjusted, enc)

        # 4. 拼接编码和增强后的解码特征
        return torch.cat([enc, dec_enhanced], dim=1)


class DenseGRU(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.grn = FeatureGRU(in_c)

    def forward(self, *features):
        concated = torch.cat(features, dim=1)
        return self.grn(concated)


class DynamicNestedUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        # ================= 动态融合组件 =================
        self.tile_buffers = nn.ModuleList([
            TileBuffer(feat_dim=nb_filter[i])
            for i in range(5)
        ])


        self.edge_concat = nn.ModuleList([
            EdgeEnhancedConcat(nb_filter[0], nb_filter[1]),  # L0: enc=16, dec=32
            EdgeEnhancedConcat(nb_filter[1], nb_filter[2]),  # L1: enc=32, dec=64
            EdgeEnhancedConcat(nb_filter[2], nb_filter[3]),  # L2: enc=64, dec=128
            EdgeEnhancedConcat(nb_filter[3], nb_filter[4])  # L3: enc=128, dec=256
        ])

        self.grn = nn.ModuleList([
            DenseGRU(nb_filter[0] * 2 + nb_filter[1]),  # Level 0
            DenseGRU(nb_filter[1] * 2 + nb_filter[2]),  # Level 1
            DenseGRU(nb_filter[2] * 2 + nb_filter[3]),  # Level 2
            DenseGRU(nb_filter[3] * 2 + nb_filter[4])  # Level 3
        ])
        self.SRAM = SpectralAdaptiveConv(num_channels=4, embed_dim=4)
        self.UpSample = nn.Upsample(
            scale_factor=2.995,
            mode='nearest'  # 上采样模式
        )


        # 定义通道调整卷积层
        self.channel_adjustments = nn.ModuleList([
            nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1),  # 从32通道调整到16通道
            nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1),  # 从64通道调整到32通道
            nn.Conv2d(nb_filter[3], nb_filter[2], kernel_size=1),  # 从128通道调整到64通道
            nn.Conv2d(nb_filter[4], nb_filter[3], kernel_size=1)  # 从256通道调整到128通道
        ])

        # ================= 编码器路径 =================
        self.pool = nn.MaxPool2d(2, 2)
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv0_0 = self._make_encoder_block(input_channels, nb_filter[0])
        self.conv1_0 = self._make_encoder_block(nb_filter[0], nb_filter[1])
        self.conv2_0 = self._make_encoder_block(nb_filter[1], nb_filter[2])
        self.conv3_0 = self._make_encoder_block(nb_filter[2], nb_filter[3])
        self.conv4_0 = self._make_encoder_block(nb_filter[3], nb_filter[4])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ================= 解码器路径 =================
        # 层级3解码 (x4_0 -> x3_1)
        self.conv3_1 = DynamicVGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3])

        # 层级2解码 (x3_1 -> x2_1, x2_2)
        self.conv2_1 = DynamicVGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2])  # 64+128=192 -> 64
        self.conv2_2 = DynamicVGGBlock(nb_filter[2] * 2, nb_filter[2])  # 64+64=128 -> 64

        self.conv1_1 = DynamicVGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1])  # 输入96(32+64) → 输出32
        self.conv1_2 = DynamicVGGBlock(nb_filter[1] * 2, nb_filter[1])  # 输入64(32+32) → 输出32
        self.conv1_3 = DynamicVGGBlock(nb_filter[1] * 3, nb_filter[1])  # 输入96 → 输出32

        # 层级0解码 (x1_3 -> x0_1, x0_2, x0_3, x0_4)
        self.conv0_1 = DynamicVGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv0_2 = DynamicVGGBlock(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv0_3 = DynamicVGGBlock(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv0_4 = DynamicVGGBlock(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])

        # ================= 输出层 =================
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

        self._init_weights()

    def _make_encoder_block(self, in_c, out_c):
        return nn.Sequential(
            DynamicVGGBlock(in_c, out_c),
            self.pool
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _update_buffer(self, level, coords, feat):
        """更新指定层级的缓冲区"""
        self.tile_buffers[level].update_buffer(coords, feat)



    def forward(self, x, λ,tile_coords=(0, 0)):
        # ================= 编码阶段 =================
        x = self.SRAM(x, λ)
        # print("SRAM output shape:", x.shape)  # 例如 [B, 4, 128, 128]
        # x = F.interpolate(
        #     x,
        #     size=(256, 256),  # 直接指定目标尺寸
        #     mode='bilinear',  # 或 'nearest'/'bicubic'
        #     align_corners=False  # 当 mode='bilinear' 时需设置
        # )
        x = self.UpSample(x)

        #print('input:', x.shape)
        # 层级0
        x0_0 = self.conv0_0(self.up(x))
        #print('x0_0:', x0_0.shape)
        self._update_buffer(0, tile_coords, x0_0)
        #print('x0_0:', x0_0.shape)

        # 层级1
        x1_0 = self.conv1_0(x0_0)
        self._update_buffer(1, tile_coords, x1_0)
        #print('x1_0:', x1_0.shape)

        # 层级2
        x2_0 = self.conv2_0(x1_0)
        self._update_buffer(2, tile_coords, x2_0)
        #print('x2_0:', x2_0.shape)

        # 层级3
        x3_0 = self.conv3_0(x2_0)
        self._update_buffer(3, tile_coords, x3_0)
        #print('x3_0:', x3_0.shape)

        # 层级4
        x4_0 = self.conv4_0(x3_0)
        self._update_buffer(4, tile_coords, x4_0)
        #print('x4_0:', x4_0.shape)

        # ================= 解码阶段 =================

        # 层级4解码 (x4_0 -> x3_1)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        #print('x3_1:', x3_1.shape)

        # 层级3解码 (x3_1 -> x2_1, x2_2)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1], 1))
        #print('x2_1:', x2_1.shape)
        #print('x2_2:', x2_2.shape)

        # 层级2解码 (x2_2 -> x1_1, x1_2, x1_3)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_2)], 1))
        #print('x1_1:', x1_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1], 1))
        #print('x1_2:', x1_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2], 1))
        #print('x1_3:', x1_3.shape)

        # 层级1解码 (x1_3 -> x0_1, x0_2, x0_3, x0_4)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_3)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_3)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_3)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_1:', x0_1.shape)
        # print('x0_2:', x0_2.shape)
        # print('x0_3:', x0_3.shape)
        # print('x0_4:', x0_4.shape)

        # ================= 输出 =================

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output



class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#scale_factor:放大的倍数  插值

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        # print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from SFF.MDAF import MDAF
from SFF.FMS import FMS


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
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
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class WS(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WS, self).__init__()
        self.pre_conv = Conv(in_channels, in_channels, kernel_size=1)
        self.pre_conv2 = Conv(in_channels, in_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(in_channels, decode_channels, kernel_size=3)

    def forward(self, x, res, ade):
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x + fuse_weights[2] * ade
        x = self.post_conv(x)
        return x


class AuxHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


class SFFNet(nn.Module):
    def __init__(self,
                 decode_channels=96,
                 dropout=0.1,
                 backbone_name="convnext_tiny.in12k_ft_in1k_384",
                 pretrained=False,
                 window_size=8,
                 num_classes=2,
                 use_aux_loss=True
                 ):
        super().__init__()
        self.use_aux_loss = use_aux_loss
        self.backbone = timm.create_model(model_name=backbone_name, features_only=True, pretrained=pretrained,
                                          output_stride=32, out_indices=(0, 1, 2, 3), in_chans=4)

        self.conv2 = ConvBN(192, decode_channels, kernel_size=1)
        self.conv3 = ConvBN(384, decode_channels, kernel_size=1)
        self.conv4 = ConvBN(768, decode_channels, kernel_size=1)

        self.MDAF_L = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.MDAF_H = MDAF(decode_channels, num_heads=8, LayerNorm_type='WithBias')
        self.fuseFeature = FMS(in_ch=4 * decode_channels, out_ch=decode_channels, num_heads=8, window_size=window_size)
        self.WF1 = WF(in_channels=decode_channels, decode_channels=decode_channels)
        self.WF2 = WF(in_channels=decode_channels, decode_channels=decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.down = Conv(in_channels=4 * decode_channels, out_channels=decode_channels, kernel_size=1)

    def forward(self, x, imagename=None):
        # 存储所有特征图的字典
        feature_maps = {}

        b = x.size()[0]
        h, w = x.size()[-2:]

        # 获取backbone各层输出
        res1, res2, res3, res4 = self.backbone(x)
        feature_maps['backbone_res1'] = res1
        feature_maps['backbone_res2'] = res2
        feature_maps['backbone_res3'] = res3
        feature_maps['backbone_res4'] = res4

        res1h, res1w = res1.size()[-2:]

        # 转换通道数
        res2 = self.conv2(res2)
        res3 = self.conv3(res3)
        res4 = self.conv4(res4)
        feature_maps['conv_res2'] = res2
        feature_maps['conv_res3'] = res3
        feature_maps['conv_res4'] = res4

        # 上采样
        res2 = F.interpolate(res2, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res3 = F.interpolate(res3, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res4 = F.interpolate(res4, size=(res1h, res1w), mode='bicubic', align_corners=False)
        feature_maps['upsampled_res2'] = res2
        feature_maps['upsampled_res3'] = res3
        feature_maps['upsampled_res4'] = res4

        # 拼接特征
        middleres = torch.cat([res1, res2, res3, res4], dim=1)
        feature_maps['concatenated_features'] = middleres

        # 特征融合
        fusefeature_L, fusefeature_H, glb, local = self.fuseFeature(middleres, imagename)
        feature_maps['fusefeature_L'] = fusefeature_L
        feature_maps['fusefeature_H'] = fusefeature_H
        feature_maps['glb_before_MDAF'] = glb
        feature_maps['local_before_MDAF'] = local

        # MDAF处理
        glb = self.MDAF_L(fusefeature_L, glb)
        local = self.MDAF_H(fusefeature_H, local)
        feature_maps['glb_after_MDAF'] = glb
        feature_maps['local_after_MDAF'] = local

        # 第一次加权融合
        res = self.WF1(glb, local)
        feature_maps['after_WF1'] = res

        # 下采样和第二次融合
        middleres = self.down(middleres)
        feature_maps['downsampled_features'] = middleres

        res = F.interpolate(res, size=(res1h, res1w), mode='bicubic', align_corners=False)
        res = middleres + res
        feature_maps['before_WF2'] = res

        res = self.WF2(res, res1)
        feature_maps['after_WF2'] = res

        # 分割头
        res = self.segmentation_head(res)
        feature_maps['segmentation_output'] = res

        # 最终输出
        # x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
        # feature_maps['final_output'] = x
        #
        # if self.training:
        #     if self.use_aux_loss == True:
        #         return x, feature_maps
        #     else:
        #         return x, feature_maps
        # else:
        #     return x, feature_maps
        if self.training:
            if self.use_aux_loss == True:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                return x
            else:
                x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
                return x
        else:
            x = F.interpolate(res, size=(h, w), mode='bilinear', align_corners=False)
            return x

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)
if __name__ == '__main__':

    x = torch.randn(1, 4, 1024, 1024)

    model=DynamicNestedUNet(num_classes=2, input_channels=4)
    model=model.cuda()
    # flops, params = ptflops.get_model_complexity_info(net, (3, 128, 128), as_strings=True,
    #                                                   print_per_layer_stat=True, verbose=True)
    # print('FLOPs:  ' + flops)
    # print('Params: ' + params)
    num_channels = 4
    start_wavelength = 450  # 起始波长 (nm)
    end_wavelength = 900  # 终止波长 (nm)
    batch_size = 1  # 批量大小

    # 生成单个样本的波长向量
    λ_single = torch.linspace(
        start_wavelength,
        end_wavelength,
        num_channels,
        dtype=torch.float32,
        device='cuda'  # 直接在GPU上创建张量
    )  #

    # 将波长向量扩展到批量维度
    λ_batch = λ_single.unsqueeze(0).expand(batch_size, -1)  # [B, C]

    # 输出：tensor([450.0, 480.0, 510.0, ..., 900.0])
    x=x.cuda()
    out = model(x,λ_batch)
    # out,gradient = net(x)
    print(out.shape)
    # print(gradient.shape)