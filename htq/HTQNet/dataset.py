import os
import torch
import imageio
import numpy as np
from net import HTQNet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.mask2polygons import mask_to_polygons
from torch.utils.data.dataloader import default_collate

def custom_collate_fn(batch):
    """
    自定义collate_fn，处理不规则的多边形顶点序列
    :param batch: 一个batch的数据，格式为 [(image1, target1), (image2, target2), ...]
    :return: 对齐后的图像和标签
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # 对齐图像
    images = default_collate(images)  # 使用默认的collate函数对齐图像

    # 对齐标签（保持多边形顶点序列的原始格式）
    aligned_targets = []
    for target in targets:
        aligned_target = {
            "polygons": target["polygons"],  # 多边形顶点序列保持不变
            "labels": target["labels"],     # 标签保持不变
            "boxes": target["boxes"]       # 边界框保持不变
        }
        aligned_targets.append(aligned_target)

    return images, aligned_targets

# 读取赤潮数据
class RedTideDataset(Dataset):

    # 获取存储图像和标签的文件夹路径及名称列表
    def __init__(self, image_dir, annotation_dir, transform=None, image_size=(256, 256)):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_name_list = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        self.image_size = image_size

        # 过滤掉没有多边形的样本
        self.valid_indices = []
        for idx in range(len(self.image_name_list)):
            annotation_path = os.path.join(self.annotation_dir, self.image_name_list[idx])
            annotation = imageio.imread(annotation_path)
            annotation = annotation.astype(np.float32)
            polygons = mask_to_polygons(annotation, 0)
            if len(polygons) > 0:  # 只保留有多边形的样本
                self.valid_indices.append(idx)

    # 获取训练样本的数量
    def __len__(self):
        return len(self.valid_indices)

    # 获取每一个训练样本对
    def __getitem__(self, idx):

        # 获取有效的样本索引
        valid_idx = self.valid_indices[idx]
        image_name = self.image_name_list[valid_idx]

        # 读取训练图像
        image_path = os.path.join(self.image_dir, image_name)
        image = imageio.imread(image_path)
        image = image.astype(np.float32)

        # 读取训练标签
        annotation_path = os.path.join(self.annotation_dir, image_name)
        annotation = imageio.imread(annotation_path)
        annotation = annotation.astype(np.float32)

        # 将mask转为多边形进而保存起来
        polygons = mask_to_polygons(annotation, 0)

        # 归一化顶点坐标
        normalized_polygons = []
        for polygon in polygons:
            normalized_polygon = polygon / np.array([self.image_size[1], self.image_size[0]])  # 归一化到 [0, 1] 范围
            normalized_polygons.append(normalized_polygon)

        # 计算每个多边形的边界框
        boxes = []
        for polygon in normalized_polygons:
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]
            boxes.append([x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()])

        labels = torch.ones((len(boxes),), dtype=torch.int64)

        # 保存多边形和边界框信息
        target = {
            "polygons": [torch.tensor(polygon).to(torch.float32) for polygon in normalized_polygons],
            "labels": labels,
            "boxes": torch.tensor(boxes, dtype=torch.float32)
        }

        if self.transform:
            image = self.transform(image)

        return image, target

# 使用main方法对该类进行测试
if __name__ == '__main__':
    model = HTQNet()
    dataset = RedTideDataset(image_dir='data/train/images', annotation_dir='data/train/labels', transform=transforms.ToTensor())
    print("训练集的长度为：{}".format(len(dataset)))
    dataloader = DataLoader(dataset, 2, collate_fn=custom_collate_fn)
    for imgs, labels in dataloader:
        print("图像形状：", imgs.shape)
        print("标签数量：", len(labels))
        outputs = model(imgs)
        print("预测结果：", outputs)
        for i, label in enumerate(labels):
            print(f"样本 {i} 的多边形数量：", len(label["polygons"]))
            print(f"样本 {i} 的边界框：", label["boxes"])

