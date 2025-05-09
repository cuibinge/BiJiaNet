import os

import cv2
import numpy as np
import torch
import torch.utils.data
from osgeo import gdal


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
#             |   ├── 0a7e06.jpg
#             │   ├── 0aab0a.jpg
#             │   ├── 0b1761.jpg
#             │   ├── ...
#             |
#             └── masks
#                 ├── 0
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 |
#                 ├── 1
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 ...
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
#         img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
#
#         mask = []
#         for i in range(self.num_classes):
#             mask.append(cv2.imread(os.path.join(self.mask_dir,
#                         img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
#         mask = np.dstack(mask)
#
#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)#这个包比较方便，能把mask也一并做掉
#             img = augmented['image']#参考https://github.com/albumentations-team/albumentations
#             mask = augmented['mask']
#
#         img = img.astype('float32') / 255
#         img = img.transpose(2, 0, 1)
#         mask = mask.astype('float32') / 255
#         mask = mask.transpose(2, 0, 1)
#
#         return img, mask, {'img_id': img_id}
#


#四通道
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
#         img = img.astype('float32') / 256  # 归一化到 [0, 1]
#         img = img.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#
#         # 标签归一化并调整维度顺序
#         mask = mask.astype('float32') / 256 # 归一化到 [0, 1]
#         mask = mask.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#
#         return img, mask, {'img_id': img_id}
#     # def __getitem__(self, idx):
#     #     img_id = self.img_ids[idx]
#     #
#     #     # 加载四通道图像
#     #     img_path = os.path.join(self.img_dir, img_id + self.img_ext)
#     #     img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
#     #     if img is None:
#     #         raise ValueError(f"Failed to load image: {img_path}")
#     #
#     #     # 确保图像是四通道
#     #     if img.shape[2] != 4:
#     #         raise ValueError(f"Image {img_id} does not have 4 channels. Shape: {img.shape}")
#     #
#     #     # 加载 Mask
#     #     mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
#     #     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 加载单通道 Mask
#     #     if mask is None:
#     #         raise ValueError(f"Failed to load mask: {mask_path}")
#     #
#     #     # 将单通道 Mask 转换为多通道 Mask
#     #     mask = np.stack([mask] * self.num_classes, axis=-1)  # 复制单通道为多通道
#     #
#     #     # 确保 Mask 的通道数与类别数一致
#     #     if mask.shape[2] != self.num_classes:
#     #         raise ValueError(f"Mask {img_id} has {mask.shape[2]} channels, but expected {self.num_classes} channels.")
#     #
#     #     # 调整图像和 Mask 的大小到 target_size
#     #     # img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)  # 双线性插值
#     #     # mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)  # 最近邻插值（保证整数标签）
#     #
#     #     # 数据增强
#     #     if self.transform is not None:
#     #         augmented = self.transform(image=img, mask=mask)  # 同步增强图像和标签
#     #         img = augmented['image']
#     #         mask = augmented['mask']
#     #
#     #     # 图像归一化并调整维度顺序
#     #     img = img.astype('float32') / 256  # 归一化到 [0, 1]
#     #     img = img.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#     #
#     #     # 标签归一化并调整维度顺序
#     #     mask = mask.astype('float32') / 256  # 归一化到 [0, 1]
#     #     mask = mask.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#     #
#     #     return img, mask, {'img_id': img_id}
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


# import os
# import cv2
# import numpy as np
# import torch
# import torch.utils.data
# from osgeo import gdal  # 用于读取多通道 TIFF 文件
# from albumentations import Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Resize, Normalize, \
#     RandomBrightnessContrast, GaussNoise, GaussianBlur
#
#
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, ndvi=False,
#                  norm=False, std=False, equalize=False, min_list=None, max_list=None, mean_list=None, std_list=None):
#         """
#         自定义数据集类，用于加载多通道 TIFF 图像和对应的掩码。
#
#         Args:
#             img_ids (list): 图像 ID 列表。
#             img_dir (str): 图像文件夹路径。
#             mask_dir (str): 掩码文件夹路径。
#             img_ext (str): 图像文件扩展名。
#             mask_ext (str): 掩码文件扩展名。
#             num_classes (int): 类别数。
#             transform (callable, optional): 数据增强函数。默认为 None。
#             ndvi (bool, optional): 是否计算 NDVI。默认为 False。
#             norm (bool, optional): 是否进行归一化。默认为 False。
#             std (bool, optional): 是否进行标准化。默认为 False。
#             equalize (bool, optional): 是否进行直方图均衡化。默认为 False。
#             min_list (list, optional): 每个通道的最小值列表，用于归一化。
#             max_list (list, optional): 每个通道的最大值列表，用于归一化。
#             mean_list (list, optional): 每个通道的均值列表，用于标准化。
#             std_list (list, optional): 每个通道的标准差列表，用于标准化。
#         """
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
#         self.num_classes = num_classes
#         self.transform = transform
#         self.ndvi = ndvi
#         self.norm = norm
#         self.std = std
#         self.equalize = equalize
#
#         self.min_list = min_list if min_list is not None else [0, 0, 0, 0]
#         self.max_list = max_list if max_list is not None else [1, 1, 1, 1]
#         self.mean_list = mean_list if mean_list is not None else [0, 0, 0, 0]
#         self.std_list = std_list if std_list is not None else [1, 1, 1, 1]
#
#         if self.norm and (self.min_list is None or self.max_list is None):
#             raise ValueError("min_list and max_list must be provided when norm is True")
#
#         if self.std and (self.mean_list is None or self.std_list is None):
#             raise ValueError("mean_list and std_list must be provided when std is True")
#
#     def __len__(self):
#         return len(self.img_ids)
#
#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#
#         # 加载多通道图像
#         img_path = os.path.join(self.img_dir, img_id + self.img_ext)
#         img = self._load_multichannel_tif(img_path)  # 使用自定义函数加载多通道 TIFF 文件
#         if img is None:
#             raise ValueError(f"Failed to load image: {img_path}")
#
#         # 确保图像是四通道
#         if img.shape[2] != 4:
#             raise ValueError(f"Image {img_id} does not have 4 channels. Shape: {img.shape}")
#
#         # 加载掩码
#         mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
#         mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 加载单通道掩码
#         if mask is None:
#             raise ValueError(f"Failed to load mask: {mask_path}")
#
#         # 将单通道掩码转换为多通道掩码
#         mask = np.stack([mask] * self.num_classes, axis=-1)  # 复制单通道为多通道
#
#         # 确保掩码的通道数与类别数一致
#         if mask.shape[2] != self.num_classes:
#             raise ValueError(f"Mask {img_id} has {mask.shape[2]} channels, but expected {self.num_classes} channels.")
#
#         # 分离波段
#         B1, B2, B3, B4 = cv2.split(img)
#
#         # 处理 NaN 值
#         B1 = np.nan_to_num(B1)
#         B2 = np.nan_to_num(B2)
#         B3 = np.nan_to_num(B3)
#         B4 = np.nan_to_num(B4)
#
#         # 直方图均衡化
#         if self.equalize:
#             B1 = cv2.equalizeHist(B1.astype(np.uint8))
#             B2 = cv2.equalizeHist(B2.astype(np.uint8))
#             B3 = cv2.equalizeHist(B3.astype(np.uint8))
#             B4 = cv2.equalizeHist(B4.astype(np.uint8))
#
#         # 计算 NDVI
#         if self.ndvi:
#             NDVI = (B4 - B3) / (B4 + B3 + 1e-6)
#             NDVI = np.clip(NDVI, -1, 1)  # 限制 NDVI 值范围
#             NDVI = ((NDVI + 1) / 2).astype(np.float32)  # 归一化到 [0, 1]
#
#         # 归一化
#         if self.norm:
#             B1 = (B1 - self.min_list[0]) / (self.max_list[0] - self.min_list[0])
#             B2 = (B2 - self.min_list[1]) / (self.max_list[1] - self.min_list[1])
#             B3 = (B3 - self.min_list[2]) / (self.max_list[2] - self.min_list[2])
#             B4 = (B4 - self.min_list[3]) / (self.max_list[3] - self.min_list[3])
#
#         # 标准化
#         if self.std:
#             B1 = (B1 - self.mean_list[0]) / (self.std_list[0] + 1e-6)
#             B2 = (B2 - self.mean_list[1]) / (self.std_list[1] + 1e-6)
#             B3 = (B3 - self.mean_list[2]) / (self.std_list[2] + 1e-6)
#             B4 = (B4 - self.mean_list[3]) / (self.std_list[3] + 1e-6)
#
#         # 合并波段
#         if self.ndvi:
#             img = cv2.merge([B1, B2, B3, NDVI])
#         else:
#             img = cv2.merge([B1, B2, B3, B4])
#
#         # 数据增强
#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)  # 同步增强图像和标签
#             img = augmented['image']
#             mask = augmented['mask']
#
#         # 图像归一化并调整维度顺序
#         img = img.astype('float32') / 256  # 归一化到 [0, 1]
#         img = img.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#
#         # 掩码归一化并调整维度顺序
#         mask = mask.astype('float32') / 256  # 归一化到 [0, 1]
#         mask = mask.transpose(2, 0, 1)  # 将通道维度移到最前面 (H, W, C) -> (C, H, W)
#
#         return img, mask, {'img_id': img_id}
#
#     def _load_multichannel_tif(self, path):
#         """
#         加载多通道 TIFF 文件。
#
#         Args:
#             path (str): TIFF 文件路径。
#
#         Returns:
#             np.ndarray: 多通道图像，形状为 (H, W, C)。
#         """
#         dataset = gdal.Open(path)
#         if dataset is None:
#             raise ValueError(f"Failed to load TIFF file: {path}")
#
#         # 读取所有通道
#         channels = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
#
#         # 将通道堆叠在一起
#         img = np.stack(channels, axis=-1)
#         return img
import os
import cv2
import numpy as np
import torch
import torch.utils.data
from osgeo import gdal
from albumentations import Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Resize, Normalize, \
    RandomBrightnessContrast, GaussNoise, GaussianBlur


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None, ndvi=False,
                 norm=False, std=False, equalize=False, min_list=None, max_list=None, mean_list=None, std_list=None):
        """
        自定义数据集类，用于加载多通道 TIFF 图像和对应的掩码。

        Args:
            img_ids (list): 图像 ID 列表。
            img_dir (str): 图像文件夹路径。
            mask_dir (str): 掩码文件夹路径。
            img_ext (str): 图像文件扩展名。
            mask_ext (str): 掩码文件扩展名。
            num_classes (int): 类别数。
            transform (callable, optional): 数据增强函数。默认为 None。
            ndvi (bool, optional): 是否计算 NDVI。默认为 False。
            norm (bool, optional): 是否进行归一化。默认为 False。
            std (bool, optional): 是否进行标准化。默认为 False。
            equalize (bool, optional): 是否进行直方图均衡化。默认为 False。
            min_list (list, optional): 每个通道的最小值列表，用于归一化。
            max_list (list, optional): 每个通道的最大值列表，用于归一化。
            mean_list (list, optional): 每个通道的均值列表，用于标准化。
            std_list (list, optional): 每个通道的标准差列表，用于标准化。
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.ndvi = ndvi
        self.norm = norm
        self.std = std
        self.equalize = equalize

        self.min_list = min_list if min_list is not None else [0, 0, 0, 0]
        self.max_list = max_list if max_list is not None else [1, 1, 1, 1]
        self.mean_list = mean_list if mean_list is not None else [0, 0, 0, 0]
        self.std_list = std_list if std_list is not None else [1, 1, 1, 1]

        if self.norm and (self.min_list is None or self.max_list is None):
            raise ValueError("min_list and max_list must be provided when norm is True")

        if self.std and (self.mean_list is None or self.std_list is None):
            raise ValueError("mean_list and std_list must be provided when std is True")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        # 加载多通道图像和波长信息
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img, wavelengths = self._load_multichannel_tif_with_wavelength(img_path)  # 修改后的加载函数
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        # 确保图像是四通道
        if img.shape[2] != 4:
            raise ValueError(f"Image {img_id} does not have 4 channels. Shape: {img.shape}")

        # 确保波长信息存在且与通道数匹配
        if wavelengths is None or len(wavelengths) != img.shape[2]:
            wavelengths = [0.0] * img.shape[2]  # 默认值

        # 加载掩码
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # 将单通道掩码转换为多通道掩码
        mask = np.stack([mask] * self.num_classes, axis=-1)

        # 确保掩码的通道数与类别数一致
        if mask.shape[2] != self.num_classes:
            raise ValueError(f"Mask {img_id} has {mask.shape[2]} channels, but expected {self.num_classes} channels.")

        # 分离波段
        B1, B2, B3, B4 = cv2.split(img)

        # 处理 NaN 值
        B1 = np.nan_to_num(B1)
        B2 = np.nan_to_num(B2)
        B3 = np.nan_to_num(B3)
        B4 = np.nan_to_num(B4)

        # 直方图均衡化
        if self.equalize:
            B1 = cv2.equalizeHist(B1.astype(np.uint8))
            B2 = cv2.equalizeHist(B2.astype(np.uint8))
            B3 = cv2.equalizeHist(B3.astype(np.uint8))
            B4 = cv2.equalizeHist(B4.astype(np.uint8))

        # 计算 NDVI
        if self.ndvi:
            NDVI = (B4 - B3) / (B4 + B3 + 1e-6)
            NDVI = np.clip(NDVI, -1, 1)
            NDVI = ((NDVI + 1) / 2).astype(np.float32)

        # 归一化
        if self.norm:
            B1 = (B1 - self.min_list[0]) / (self.max_list[0] - self.min_list[0])
            B2 = (B2 - self.min_list[1]) / (self.max_list[1] - self.min_list[1])
            B3 = (B3 - self.min_list[2]) / (self.max_list[2] - self.min_list[2])
            B4 = (B4 - self.min_list[3]) / (self.max_list[3] - self.min_list[3])

        # 标准化
        if self.std:
            B1 = (B1 - self.mean_list[0]) / (self.std_list[0] + 1e-6)
            B2 = (B2 - self.mean_list[1]) / (self.std_list[1] + 1e-6)
            B3 = (B3 - self.mean_list[2]) / (self.std_list[2] + 1e-6)
            B4 = (B4 - self.mean_list[3]) / (self.std_list[3] + 1e-6)

        # 合并波段
        if self.ndvi:
            img = cv2.merge([B1, B2, B3, NDVI])
        else:
            img = cv2.merge([B1, B2, B3, B4])

        # 数据增强
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 图像归一化并调整维度顺序
        img = img.astype('float32') / 256
        img = img.transpose(2, 0, 1)

        # 掩码归一化并调整维度顺序
        mask = mask.astype('float32') / 256
        mask = mask.transpose(2, 0, 1)

        # 将波长信息转换为张量
        wavelengths_tensor = torch.tensor(wavelengths, dtype=torch.float32)

        return img, mask, {'img_id': img_id, 'wavelengths': wavelengths_tensor}

    def _load_multichannel_tif_with_wavelength(self, path):
        """
        加载多通道 TIFF 文件及其波长信息。

        Args:
            path (str): TIFF 文件路径。

        Returns:
            tuple: (多通道图像, 波长列表)
        """
        dataset = gdal.Open(path)
        if dataset is None:
            raise ValueError(f"Failed to load TIFF file: {path}")

        # 读取所有通道
        channels = [dataset.GetRasterBand(i + 1).ReadAsArray() for i in range(dataset.RasterCount)]
        img = np.stack(channels, axis=-1)

        # 尝试获取波长信息
        wavelengths = None
        metadata = dataset.GetMetadata()

        # 方法1: 检查ENVI格式的波长元数据
        if 'ENVI' in metadata:
            envi_meta = metadata['ENVI']
            if 'wavelength' in envi_meta:
                wavelengths = list(map(float, envi_meta['wavelength'].strip('{}').split(',')))

        # 方法2: 检查波段描述中的波长信息
        if wavelengths is None:
            wavelengths = []
            for i in range(1, dataset.RasterCount + 1):
                band = dataset.GetRasterBand(i)
                desc = band.GetDescription()
                if desc and 'nm' in desc:
                    try:
                        wavelength = float(desc.split('nm')[0].strip())
                        wavelengths.append(wavelength)
                    except ValueError:
                        wavelengths.append(0.0)
                else:
                    wavelengths.append(0.0)

        # 方法3: 如果没有找到波长信息，使用默认值
        if not wavelengths or len(wavelengths) != dataset.RasterCount:
            wavelengths = [0.0] * dataset.RasterCount

        return img, wavelengths