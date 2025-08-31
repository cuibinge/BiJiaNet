from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import imageio
from imageio.v2 import imread
import os

def process_image(image_path, ndvi=False, norm=False, std=False, equalize=False,
                  min_list=None, max_list=None, mean_list=None, std_list=None):
    """
    处理图像的函数，支持多种处理方式，包括NDVI计算、标准化、均衡化等。

    参数：
    - img_data: 输入的图像数据，格式为 numpy array (H, W, 4)，即包含4个波段的图像
    - ndvi: 是否计算NDVI
    - norm: 是否进行归一化
    - std: 是否进行标准化
    - equalize: 是否进行直方图均衡化
    - min_list: 归一化时的最小值列表
    - max_list: 归一化时的最大值列表
    - mean_list: 标准化时的均值列表
    - std_list: 标准化时的标准差列表

    返回：
    - 处理后的图像（numpy array）
    """

    if norm and (min_list is None or max_list is None):
        raise Exception("Unexpected max_list and min_list when norm is True")


    if std and (mean_list is None or std_list is None):
        raise Exception("Unexpected mean_list and std_list when std is True")

    image_data = imread(image_path)
    # 分离波段
    B1, B2, B3, B4 = cv2.split(image_data)

    # 直方图均衡化
    if equalize:
        B1 = cv2.equalizeHist(B1)
        B2 = cv2.equalizeHist(B2)
        B3 = cv2.equalizeHist(B3)
        B4 = cv2.equalizeHist(B4)

    # 处理 NaN 值
    B1[np.isnan(B1)] = 0
    B2[np.isnan(B2)] = 0
    B3[np.isnan(B3)] = 0
    B4[np.isnan(B4)] = 0

    # 如果计算 NDVI
    if ndvi:
        B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
        B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
        B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
        NDVI = (B4 - B3) / ((B4 + B3) + 1e-6)
        NDVI[(NDVI > 1) | (NDVI < -1)] = 0

        NDVI_normalization = ((NDVI - np.min(NDVI)) / (np.max(NDVI) - np.min(NDVI))).astype('float32')
        img_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization, NDVI_normalization])
        img_data[np.isnan(img_data)] = 0
    # 如果进行归一化
    elif norm:
        B1_normalization = ((B1 - min_list[0]) / (max_list[0] - min_list[0]) * 1).astype('float32')
        B2_normalization = ((B2 - min_list[1]) / (max_list[1] - min_list[1]) * 1).astype('float32')
        B3_normalization = ((B3 - min_list[2]) / (max_list[2] - min_list[2]) * 1).astype('float32')
        B4_normalization = ((B4 - min_list[3]) / (max_list[3] - min_list[3]) * 1).astype('float32')
        img_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization])
        img_data[np.isnan(img_data)] = 0
    # 如果进行标准化
    elif std:
        B1_standardization = ((B1 - mean_list[0]) / (std_list[0] + 1e-6)).astype('float32')
        B2_standardization = ((B2 - mean_list[1]) / (std_list[1] + 1e-6)).astype('float32')
        B3_standardization = ((B3 - mean_list[2]) / (std_list[2] + 1e-6)).astype('float32')
        B4_standardization = ((B4 - mean_list[3]) / (std_list[3] + 1e-6)).astype('float32')
        img_data = cv2.merge([B1_standardization, B2_standardization, B3_standardization, B4_standardization])
        img_data[np.isnan(img_data)] = 0
    else:
        img_data = cv2.merge([B1, B2, B3, B4])

    image = np.array(img_data).astype('float32')

    return image

# ===== 固定预处理配置 =====
NDVI = False
NORM = True  # 归一化
STD  = True   # 标准化
EQUALIZE = False

MAX_LIST = [840, 996, 729, 489]
MIN_LIST = [0, 220, 0, 0]
MEAN_LIST = [358.56197836, 278.51276384, 146.58589797, 48.23313843]
STD_LIST  = [28.13160621, 41.51963463, 18.1556377, 9.82935977]
# ==========================

def _process_array(img: np.ndarray) -> np.ndarray:
    """按固定配置做预处理（此刻等价于 float32）。"""
    if EQUALIZE:
        b = cv2.split(img)
        b = [cv2.equalizeHist(x) for x in b]
        img = cv2.merge(b)

    # 归一化
    if NORM:
        B1n = ((img[:, :, 0] - MIN_LIST[0]) / (MAX_LIST[0] - MIN_LIST[0])).astype('float32')
        B2n = ((img[:, :, 1] - MIN_LIST[1]) / (MAX_LIST[1] - MIN_LIST[1])).astype('float32')
        B3n = ((img[:, :, 2] - MIN_LIST[2]) / (MAX_LIST[2] - MIN_LIST[2])).astype('float32')
        B4n = ((img[:, :, 3] - MIN_LIST[3]) / (MAX_LIST[3] - MIN_LIST[3])).astype('float32')
        img = cv2.merge([B1n, B2n, B3n, B4n])

    # 标准化
    if STD:
        B1s = ((img[:, :, 0] - MEAN_LIST[0]) / (STD_LIST[0] + 1e-6)).astype('float32')
        B2s = ((img[:, :, 1] - MEAN_LIST[1]) / (STD_LIST[1] + 1e-6)).astype('float32')
        B3s = ((img[:, :, 2] - MEAN_LIST[2]) / (STD_LIST[2] + 1e-6)).astype('float32')
        B4s = ((img[:, :, 3] - MEAN_LIST[3]) / (STD_LIST[3] + 1e-6)).astype('float32')
        img = cv2.merge([B1s, B2s, B3s, B4s])

    return img.astype('float32')

def _reflect_pad_crop(arr, x1, y1, x2, y2, H, W):
    need_pad = (x1 < 0) or (y1 < 0) or (x2 > W) or (y2 > H)
    if not need_pad:
        return arr[y1:y2, x1:x2, ...].copy()
    top = max(0, -y1); left = max(0, -x1)
    bottom = max(0, y2 - H); right = max(0, x2 - W)
    if arr.ndim == 3:
        padw = ((top,bottom),(left,right),(0,0))
    else:
        padw = ((top,bottom),(left,right))
    padded = np.pad(arr, padw, mode="reflect")
    return padded[y1+top:y2+top, x1+left:x2+left, ...].copy()

def get_tiles_from_rc(
    big_tif_path: str,
    row_idx: int, col_idx: int,
    block_size: int = 256,
    stride: int = 256,
    halo_scale: float = 1.25,
    use_diagonals: bool = True,
):
    """
    给 (row_idx, col_idx) + 大图路径，返回中心tile/halo/邻居（已预处理）。
    """
    img = imageio.imread(big_tif_path)
    if img.ndim == 2:  # 单通道时扩一维
        img = img[..., None]
    H, W, C = img.shape

    # 行列号 → 像素坐标（从1开始）
    x0 = (col_idx - 1) * stride
    y0 = (row_idx - 1) * stride

    # 中心块
    center_raw = _reflect_pad_crop(img, x0, y0, x0+block_size, y0+block_size, H, W)
    center = _process_array(center_raw)

    # Halo 扩张块
    T_big = int(round(block_size * halo_scale))
    halo_pad = (T_big - block_size) // 2
    halo_raw = _reflect_pad_crop(img,
                                 x0 - halo_pad, y0 - halo_pad,
                                 x0 + block_size + halo_pad,
                                 y0 + block_size + halo_pad, H, W)
    # 可能存在边界导致尺寸不完全等于 T_big×T_big，做最近邻对齐
    if halo_raw.shape[0] != T_big or halo_raw.shape[1] != T_big:
        yy = np.linspace(0, halo_raw.shape[0]-1, T_big).round().astype(int)
        xx = np.linspace(0, halo_raw.shape[1]-1, T_big).round().astype(int)
        halo_raw = halo_raw[yy][:, xx]
    halo = _process_array(halo_raw)

    # 邻居
    off4 = {"N":(0,-stride), "S":(0,stride), "W":(-stride,0), "E":(stride,0)}
    off8 = {**off4, "NW":(-stride,-stride), "NE":(stride,-stride),
            "SW":(-stride,stride), "SE":(stride,stride)}
    OFFS = off8 if use_diagonals else off4

    neighbors = {}
    for k,(dx,dy) in OFFS.items():
        n_raw = _reflect_pad_crop(img,
                                  x0+dx, y0+dy,
                                  x0+dx+block_size, y0+dy+block_size, H, W)
        neighbors[k] = _process_array(n_raw)

    return {
        "center": center,          # [T,T,C], float32
        "halo": halo,              # [T_big,T_big,C], float32
        "neighbors": neighbors,    # dict of [T,T,C], float32
        "meta": {
            "row_idx": row_idx, "col_idx": col_idx,
            "x0": x0, "y0": y0,
            "block_size": block_size,
            "stride": stride,
            "halo_pad": halo_pad,
            "T_big": T_big
        }
    }

# —— 瓦片级 GAT（不依赖外部库）——
class TileGAT(nn.Module):
    def __init__(self, dim, heads=4, relpos_dim=16):
        super().__init__()
        self.heads = heads
        self.W = nn.ModuleList([nn.Linear(dim, dim//heads, bias=False) for _ in range(heads)])
        self.a = nn.ParameterList([nn.Parameter(torch.randn(dim//heads*2 + relpos_dim)) for _ in range(heads)])
        self.rel = nn.Sequential(nn.Linear(2, relpos_dim), nn.ReLU(), nn.Linear(relpos_dim, relpos_dim))
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, z, edges, rel_offsets):
        """
        z: [N, dim]   (N个瓦片token，按一幅图或一个batch内分组处理)
        edges: List[(i,j)]  邻接边索引
        rel_offsets: Tensor[|E|, 2]  每条边(i,j)的相对位移 (dx,dy) 归一化到[-1,1]
        """
        N, dim = z.shape
        out = []
        for h in range(self.heads):
            zh = self.W[h](z)           # [N, d_h]
            d_h = zh.size(-1)
            # 聚合邻接边
            e_src = torch.tensor([i for i,j in edges], device=z.device)
            e_dst = torch.tensor([j for i,j in edges], device=z.device)
            rel = self.rel(rel_offsets)  # [E, r]

            # 打分
            cat_ij = torch.cat([zh[e_src], zh[e_dst], rel], dim=-1)  # [E, 2*d_h + r]
            e_ij = F.leaky_relu((cat_ij * self.a[h]).sum(-1), 0.2)   # [E]

            # softmax 按目的节点归一化
            alpha = torch.zeros_like(e_ij)
            # 计算每个dst的归一化（简洁写法：scatter_logsumexp也可）
            exp_e = torch.exp(e_ij)
            denom = torch.zeros(N, device=z.device).index_add_(0, e_dst, exp_e)
            alpha = exp_e / (denom[e_dst] + 1e-6)

            # 消息传递
            msg = zh[e_src] * alpha.unsqueeze(-1)  # [E, d_h]
            agg = torch.zeros(N, d_h, device=z.device).index_add_(0, e_dst, msg)  # [N, d_h]
            out.append(agg)

        z_new = torch.cat(out, dim=-1)   # [N, dim]
        z_new = self.proj_out(z_new) + z # 残差
        return z_new

# —— 边带跨窗注意力（方向对齐，条带计算）——
class EdgeCrossAttention(nn.Module):
    def __init__(self, in_dim, heads=2):
        super().__init__()
        self.heads = heads
        self.q = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.k = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.v = nn.Conv2d(in_dim, in_dim, 1, bias=False)
        self.proj = nn.Conv2d(in_dim, in_dim, 1)

    def attend_strip(self, Qi, Kj, Vj):
        # Qi/Kj/Vj: [B,C,b,L] or [B,C,L,b]  (条带维度在倒数第二)
        B, C, b, L = Qi.shape
        d = C // self.heads
        def split(x):  # [B,H,d,b,L]
            return x.view(B, self.heads, d, b, L)
        Qh, Kh, Vh = split(Qi), split(Kj), split(Vj)
        # 重排方便做注意力：把(b)和(L)合成序列
        Qv = Qh.permute(0,1,3,4,2).reshape(B,self.heads,b*L,d)     # [B,H,QL,d]
        Kv = Kh.permute(0,1,3,4,2).reshape(B,self.heads,b*L,d)
        Vv = Vh.permute(0,1,3,4,2).reshape(B,self.heads,b*L,d)
        attn = torch.matmul(Qv, Kv.transpose(-1,-2)) / (d**0.5)     # [B,H,QL,KL]
        attn = attn.softmax(-1)
        out = torch.matmul(attn, Vv)                                # [B,H,QL,d]
        out = out.view(B,self.heads,b,L,d).permute(0,1,4,2,3).reshape(B,C,b,L)
        return out

    def forward(self, Fi, Fj, b=8, direction='N'):
        # 取条带（示例：北-N，对应邻居的南-S）
        if direction=='N':
            Qi = self.q(Fi)[:,:, :b, :]   # [B,C,b,W]
            Kj = self.k(Fj)[:,:, -b:, :]
            Vj = self.v(Fj)[:,:, -b:, :]
            delta = self.attend_strip(Qi, Kj, Vj)
            Fi[:,:, :b, :] = Fi[:,:, :b, :] + self.proj(delta)
        elif direction == 'S':
            Qi = self.q(Fi)[:, :, -b:, :]
            Kj = self.k(Fj)[:, :, :b, :]
            Vj = self.v(Fj)[:, :, :b, :]
            Fi[:, :, -b:, :] = Fi[:, :, -b:, :] + self.proj(self.attend_strip(Qi, Kj, Vj))
        elif direction == 'W':
            Qi = self.q(Fi)[:, :, :, :b]  # [B,C,H,b]
            Kj = self.k(Fj)[:, :, :, -b:]
            Vj = self.v(Fj)[:, :, :, -b:]
            # 为复用 attend_strip，先把维度转成 [B,C,b,L] 形式
            Qi = Qi.permute(0, 1, 3, 2)  # -> [B,C,b,H]
            Kj = Kj.permute(0, 1, 3, 2)
            Vj = Vj.permute(0, 1, 3, 2)
            delta = self.attend_strip(Qi, Kj, Vj)  # [B,C,b,H]
            delta = delta.permute(0, 1, 3, 2)  # -> [B,C,H,b]
            Fi[:, :, :, :b] = Fi[:, :, :, :b] + self.proj(delta)
        elif direction == 'E':
            Qi = self.q(Fi)[:, :, :, :-b]  # [B,C,H,b]
            Kj = self.k(Fj)[:, :, :, b:]
            Vj = self.v(Fj)[:, :, :, b:]
            # 为复用 attend_strip，先把维度转成 [B,C,b,L] 形式
            Qi = Qi.permute(0, 1, 3, 2)  # -> [B,C,b,H]
            Kj = Kj.permute(0, 1, 3, 2)
            Vj = Vj.permute(0, 1, 3, 2)
            delta = self.attend_strip(Qi, Kj, Vj)  # [B,C,b,H]
            delta = delta.permute(0, 1, 3, 2)  # -> [B,C,H,b]
            Fi[:, :, :, :b] = Fi[:, :, :, :b] + self.proj(delta)
        return Fi


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
        #print('input:',input.shape)
        x0_0 = self.conv0_0(input)
        #print('x0_0:',x0_0.shape)
        x1_0 = self.conv1_0(self.pool(x0_0))
        #print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        #print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))
        #print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        #print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        #print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))
        #print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        #print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        #print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        #print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))
        #print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        #print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        #print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        #print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        #print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output
