import os
import json
import torch
import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class RedTideDataset(Dataset):
    def __init__(self, image_dir, json_dir, transform=None):
        """
            :param image_dir: 图像文件夹路径
            :param json_dir: JSON 标签文件夹路径
            :param transform: 数据增强（可选）
            :param max_vertices: 每个多边形的最大顶点数量（用于填充）
        """
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.transform = transform
        self.max_vertices = 0
        self.image_names = os.listdir(image_dir)

        # 筛选出 polygons 不为空的样本
        self.valid_indices = []
        for idx, image_name in enumerate(self.image_names):
            json_name = os.path.splitext(image_name)[0] + ".json"
            json_path = os.path.join(self.json_dir, json_name)
            with open(json_path, "r") as f:
                labels = json.load(f)
            # 检查 polygons 是否为空
            if len(labels["polygons"]) > 0:
                self.valid_indices.append(idx)
                for polygon in labels["polygons"]:
                    self.max_vertices = max(self.max_vertices, len(polygon["polygon"]))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):

        # 读取训练图像
        valid_idx = self.valid_indices[idx]
        image_name = self.image_names[valid_idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = imageio.imread(image_path)

        # 读取JSON标签
        json_name = os.path.splitext(image_name)[0] + ".json"
        json_path = os.path.join(self.json_dir, json_name)
        with open(json_path, "r") as f:
            labels = json.load(f)

        # 解析目标检测标签
        boxes = []
        polygons = []
        for label in labels["polygons"]:

            # 边界框 (x_min, y_min, x_max, y_max)
            bbox = label["bbox"]
            boxes.append(bbox)

            # 多边形顶点序列
            polygon = label["polygon"]
            polygons.append(polygon)

        # 边界框转换为Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 目标检测标签
        num_objects = len(boxes)
        labels = torch.ones((num_objects,), dtype=torch.int64)  # 假设所有目标都是赤潮（类别 1）

        # 将 polygons 转换为张量形式
        mask = torch.zeros((num_objects, self.max_vertices), dtype=torch.bool)
        polygons_tensor = torch.zeros((num_objects, self.max_vertices, 2), dtype=torch.float32)
        for i, polygon in enumerate(polygons):
            num_vertices = len(polygon)
            if num_vertices > self.max_vertices:
                # 如果顶点数量超过最大值，截断
                polygon = polygon[:self.max_vertices]
            polygons_tensor[i, :num_vertices] = torch.as_tensor(polygon, dtype=torch.float32)
            mask[i, :num_vertices] = True

        # 图像转换为Tensor
        image = F.to_tensor(image)

        # 数据增强
        if self.transform:
            image = self.transform(image)

        # 返回数据
        return image, {
            "boxes": boxes,  # 边界框
            "labels": labels,  # 类别标签
            "polygons": polygons_tensor, # 多边形顶点序列
            "mask": mask  # 掩码张量
        }

if __name__ == "__main__":
    image_dir = "./data/train/images"
    json_dir = "./data/train/jsons"
    dataset = RedTideDataset(image_dir, json_dir)
    print("训练集的长度为：{}".format(len(dataset)))
    # 打印样本
    image, targets = dataset[1]
    print("图像的数据类型：", image.dtype)
    print("标签的数据类型：", type(targets))
    print("Image shape:", image.shape)
    print("Boxes:", targets["boxes"].dtype, targets["boxes"])
    print("Labels:", targets["labels"].dtype, targets["labels"])
    print("Polygons:", targets["polygons"].shape)
    print("Mask:", targets["mask"].shape)