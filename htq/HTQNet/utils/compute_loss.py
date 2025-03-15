import torch
import torch.nn as nn

def compute_loss(output, target, criterion):
    """
       计算损失
       :param output: 模型输出，形状为 (max_polygons, max_vertices, 2)
       :param target: 目标键值对
       :param criterion: 损失函数（如 nn.MSELoss）
       :return: 总损失
    """
    loss = 0
    valid_samples = 0

    # 遍历每个多边形
    for polygon in target["polygons"]:
        num_vertices = len(polygon)  # 当前多边形的顶点数量
        if num_vertices == 0:
            continue
        # 遍历模型输出的每个多边形
        for pred_polygon in output:  # pred_polygon 形状为 (max_vertices, 2)
            # 截取与目标顶点数量匹配的输出
            pred_vertices = pred_polygon[: num_vertices]  # (num_vertices, 2)
            loss += criterion(pred_vertices, polygon)
            valid_samples += 1
    if valid_samples > 0:
        loss = loss / valid_samples  # 平均损失
    return loss.to(torch.float32)