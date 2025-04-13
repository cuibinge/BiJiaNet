import torch
import math
import numpy as np
from typing import Tuple  # 新增导入
from torch.nn import functional as F

def sgn(x):
    return 1.0 if x > 0 else -1.0

def afm_cuda(lines: torch.Tensor, 
            shape_info: torch.Tensor, 
            height: int, 
            width: int) -> Tuple[torch.Tensor, torch.Tensor]:  # 修改类型注解
    """
    Python实现替代原CUDA代码的功能
    参数:
        lines: [N, 4] 线段坐标(x1,y1,x2,y2)
        shape_info: [B,4] 每个样本的线段起始/结束索引及原始尺寸
        height: 输出高度
        width: 输出宽度
    返回:
        afmap: [B,2,H,W] 特征图
        aflabel: [B,1,H,W] 标签图
    """
    device = lines.device
    batch_size = shape_info.size(0)
    
    # 初始化输出张量
    afmap = torch.zeros((batch_size, 2, height, width), 
                       dtype=torch.float32, device=device)
    aflabel = torch.zeros((batch_size, 1, height, width), 
                         dtype=torch.int, device=device)
    
    # 转换为CPU numpy数组加速计算
    lines_np = lines.cpu().numpy()
    shape_info_np = shape_info.cpu().numpy()
    
    for b in range(batch_size):
        start = shape_info_np[b, 0]
        end = shape_info_np[b, 1]
        orig_h = shape_info_np[b, 2]
        orig_w = shape_info_np[b, 3]
        
        xs = width / orig_w
        ys = height / orig_h
        
        for h in range(height):
            for w in range(width):
                px = w
                py = h
                min_dis = 1e30
                
                for i in range(start, end):
                    x1, y1, x2, y2 = lines_np[i] * [xs, ys, xs, ys]
                    
                    dx = x2 - x1
                    dy = y2 - y1
                    norm2 = dx*dx + dy*dy
                    
                    t = ((px-x1)*dx + (py-y1)*dy) / (norm2 + 1e-6)
                    t = max(0.0, min(1.0, t))
                    
                    ax = x1 + t*(x2-x1) - px
                    ay = y1 + t*(y2-y1) - py
                    dis = ax*ax + ay*ay
                    
                    if dis < min_dis:
                        min_dis = dis
                        afmap[b, 0, h, w] = -sgn(ax) * math.log(abs(ax/width) + 1e-6)
                        afmap[b, 1, h, w] = -sgn(ay) * math.log(abs(ay/height) + 1e-6)
                        aflabel[b, 0, h, w] = i - start
                        
    return afmap, aflabel

def afm(lines: torch.Tensor, 
       shape_info: torch.Tensor, 
       height: int, 
       width: int) -> Tuple[torch.Tensor, torch.Tensor]:  # 修改类型注解
    """接口函数，保持与原C++接口一致"""
    return afm_cuda(lines, shape_info, height, width)