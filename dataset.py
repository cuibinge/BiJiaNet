import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir,
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
            img = augmented['image']#参考https://github.com/albumentations-team/albumentations
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}



# #四通道
# import os
# import cv2
# import numpy as np
# import torch
# import torch.utils.data
# from osgeo import gdal  # 用于读取多通道 TIFF 文件
#
#
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
#         """
#         Args:
#             img_ids (list): Image ids.
#             img_dir: Image file directory.
#             mask_dir: Mask file directory.
#             img_ext (str): Image file extension.
#             mask_ext (str): Mask file extension.
#             num_classes (int): Number of classes.
#             transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
#
#         Note:
#             Make sure to put the files as the following structure:
#             <dataset name>
#             ├── images
#             |   ├── 0a7e06.tif
#             │   ├── 0aab0a.tif
#             │   ├── 0b1761.tif
#             │   ├── ...
#             |
#             └── masks
#                 ├── 0a7e06.tif
#                 ├── 0aab0a.tif
#                 ├── 0b1761.tif
#                 ├── ...
#         """
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
#         self.num_classes = num_classes
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#
#         # 加载四通道图像
#         img_path = os.path.join(self.img_dir, img_id + self.img_ext)
#         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#         if img is None:
#             raise ValueError(f"Failed to load image: {img_path}")
#
#         # 确保图像是四通道
#         if img.shape[2] != 4:
#             raise ValueError(f"Image {img_id} does not have 4 channels. Shape: {img.shape}")
#
#         # 加载 Mask
#         mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 加载单通道 Mask
#         if mask is None:
#             raise ValueError(f"Failed to load mask: {mask_path}")
#
#         # 将单通道 Mask 转换为多通道 Mask
#         mask = np.stack([mask] * self.num_classes, axis=-1)  # 复制单通道为多通道
#
#         # 确保 Mask 的通道数与类别数一致
#         if mask.shape[2] != self.num_classes:
#             raise ValueError(f"Mask {img_id} has {mask.shape[2]} channels, but expected {self.num_classes} channels.")
#
#         # 数据增强
#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)  # 同步增强图像和标签
#             img = augmented['image']
#             mask = augmented['mask']
#
#         # 图像归一化并调整维度顺序
#         img = img.astype('float32') / 223  # 归一化到 [0, 1]
#         img = img.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#
#         # 标签归一化并调整维度顺序
#         mask = mask.astype('float32') / 223  # 归一化到 [0, 1]
#         mask = mask.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#
#         return img, mask, {'img_id': img_id}
#
#     def _load_multichannel_tif(self, path):
#         """
#         加载多通道 TIFF 文件。
#         Args:
#             path (str): TIFF 文件路径。
#         Returns:
#             np.ndarray: 多通道图像，形状为 (H, W, C)。
#         """
#         dataset = gdal.Open(path)
#         if dataset is None:
#             raise ValueError(f"Failed to load TIFF file: {path}")
#
#         # 读取所有通道
#         channels = []
#         for i in range(dataset.RasterCount):
#             band = dataset.GetRasterBand(i + 1)  # GDAL 的波段索引从 1 开始
#             channel = band.ReadAsArray()
#             channels.append(channel)
#
#         # 将通道堆叠在一起
#         mask = np.stack(channels, axis=-1)
#         return mask