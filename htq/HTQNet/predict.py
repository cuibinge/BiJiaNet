import os
import torch
import numpy as np
import matplotlib
import torchvision
from torch import nn
from net2 import HTQNet
import matplotlib.pyplot as plt
from dataset import RedTideDataset
from shapely.geometry import Polygon
from dataset import custom_collate_fn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def denormalize_vertices(normalized_polygons, image_size):
    """
        反归一化顶点坐标
        :param normalized_polygons: 归一化后的多边形顶点列表，每个元素是一个多边形的顶点序列 (N, 2)
        :param image_size: 图像的尺寸 (H, W)
        :return: 反归一化后的多边形顶点列表
    """
    H, W = image_size
    denormalized_polygons = []
    for polygon in normalized_polygons:
        denormalized_polygon = polygon * np.array([W, H])  # 反归一化到原始范围
        denormalized_polygons.append(denormalized_polygon)
    return denormalized_polygons

def filter_valid_vertices(vertices, threshold=1e-3):
    """
       过滤无效顶点
       :param vertices: 顶点序列，形状为 (N, 2)
       :param threshold: 顶点坐标的最小有效值（小于该值的顶点将被过滤）
       :return: 过滤后的顶点序列
    """
    valid_vertices = []
    for vertex in vertices:
        if np.linalg.norm(vertex) > threshold:  # 过滤掉接近 (0, 0) 的顶点
            valid_vertices.append(vertex)
    return np.array(valid_vertices)

def vertices_to_polygon(vertices):
    """
        将顶点序列转换为多边形
        :param vertices: 顶点序列，形状为 (N, 2)
        :return: 多边形对象（shapely.geometry.Polygon）
    """
    if len(vertices) < 3:  # 至少需要 3 个顶点才能构成多边形
        return None
    return Polygon(vertices)

def plot_polygons(image, polygons):
    """
        在原始图像上绘制多个多边形
        :param image: 原始图像，形状为 (C, H, W)
        :param polygons: 多边形对象列表（shapely.geometry.Polygon）
    """

    # 设置中文字体为黑体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    image = torch.squeeze(image, 0)
    image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    H, W, _ = image.shape

    # 绘制图像
    plt.figure()
    plt.imshow(image)

    # 绘制多边形
    for polygon in polygons:
        if polygon is not None:
            exterior = np.array(polygon.exterior.coords)
            plt.plot(exterior[:, 0], exterior[:, 1], 'r-', linewidth=2)  # 绘制外环（红色）
            plt.fill(exterior[:, 0], exterior[:, 1], 'r', alpha=0.3)  # 填充多边形（红色，半透明）

    plt.title("预测的多边形")
    plt.show()

image_size = (256, 256)
weight_path = "weights/epoch5_loss0.19535546004772186.pt"
model = HTQNet()
model.load_state_dict(torch.load(weight_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = RedTideDataset(image_dir='data/train/images', annotation_dir='data/train/labels', transform=transforms.ToTensor())
image = train_dataset[1][0]
image = torch.unsqueeze(image, 0)
label = train_dataset[1][1]

with torch.no_grad():
    output = model(image)
output = torch.squeeze(output)
output_np = output.numpy()

# 反归一化顶点坐标
denormalized_polygons = denormalize_vertices(output_np, image_size)

print("反归一化前序列：", output_np)
print("反归一化后序列：", denormalized_polygons)
print("实际的顶点序列：", label["polygons"])

# 过滤无效顶点并转换为多边形
polygons = []
for polygon_vertices in denormalized_polygons:
    valid_vertices = filter_valid_vertices(polygon_vertices)
    if len(valid_vertices) >= 3:  # 至少需要 3 个顶点才能构成多边形
        polygons.append(Polygon(valid_vertices))

# 在原始图像上绘制多个多边形
plot_polygons(image, polygons)


