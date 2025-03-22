import torch
import torch.nn as nn
from shapely.geometry import Polygon

def compute_iou(output, target):
    """
        计算两个多边形之间的 IoU
        :param polygon1: 预测多边形，形状为 (N, 2)
        :param polygon2: 真实多边形，形状为 (M, 2)
        :return: IoU 值
    """

    print(output)
    print(target)

    # 将需要梯度计算的张量转换为 NumPy 数组
    output_np = output.detach().numpy()
    target_np = target["polygons"].detach().numpy()

    poly1 = Polygon(output_np)
    poly2 = Polygon(target_np)
    if not poly1.is_valid or not poly2.is_valid:
        return torch.tensor(0.0, dtype=torch.float32)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union if union > 0 else 0.0
    loss = 1 - iou
    return torch.tensor(loss, dtype=torch.float32)

def iou_loss(pred_polygons, target_polygons):
    """
        计算 IoU 损失
        :param pred_polygons: 预测的多边形列表，每个元素形状为 (N, 2)
        :param target_polygons: 真实的多边形列表，每个元素形状为 (M, 2)
        :return: IoU 损失
    """
    loss = 0
    for pred, target in zip(pred_polygons, target_polygons):
        iou = compute_iou(pred, target)
        loss += 1 - iou  # IoU 损失定义为 1 - IoU
    return loss / len(pred_polygons)

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