import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import cv2
from datetime import datetime
from osgeo import gdal
from archs import SFFNet
import math

class FeatureVisualizer:
    def __init__(self, model, img_dir, img_ext='.tif', device='cuda'):
        """
        初始化特征可视化工具
        Args:
            model: 预训练的SFFNet模型
            img_dir: 图像目录路径
            img_ext: 图像文件扩展名
            device: 计算设备
        """
        self.model = model.to(device)
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.device = device

        # 设置模型为评估模式
        self.model.eval()

    def load_tif_image(self, img_path):
        """加载多通道TIFF图像"""
        dataset = gdal.Open(img_path)
        if dataset is None:
            raise ValueError(f"无法加载TIFF文件: {img_path}")

        # 读取所有通道
        channels = []
        for i in range(dataset.RasterCount):
            band = dataset.GetRasterBand(i + 1)
            channel = band.ReadAsArray()
            channel = np.nan_to_num(channel)  # 处理NaN值
            channels.append(channel)

        # 堆叠通道并归一化
        img = np.stack(channels, axis=-1).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # 归一化到[0,1]
        return img

    def preprocess_image(self, img):
        """预处理图像为模型输入格式"""
        # 转换为Tensor并添加batch维度
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # 确保是4通道输入
        if img_tensor.shape[1] != 4:
            raise ValueError(f"输入图像需要4个通道，但得到{img_tensor.shape[1]}个通道")

        return img_tensor

    def get_feature_maps(self, img_id, layer_names=None):
        """
        获取指定图像的特征图
        Args:
            img_id: 图像ID（不带扩展名）
            layer_names: 要监控的层名称列表
        """
        # 加载图像
        img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        img = self.load_tif_image(img_path)
        input_tensor = self.preprocess_image(img)

        # 设置钩子捕获特征图
        features = {'image': img, 'image_id': img_id}

        def get_features(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    features[name] = [o.detach().cpu()[0] for o in output]
                else:
                    features[name] = output.detach().cpu()[0]

            return hook

        # 默认监控层
        if layer_names is None:
            layer_names = [
                'backbone.layer1',
                'conv2',
                'conv3',
                'conv4',
                'MDAF_L',
                'MDAF_H',
                'fuseFeature',
                'WF1',
                'WF2',
                'segmentation_head.0',
                'segmentation_head.1',
                'segmentation_head.2',
                'down',
            ]

        # 注册钩子
        hooks = []
        for name, module in self.model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(get_features(name)))

        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        return features

    @staticmethod
    def visualize(features, save_dir=None, max_channels=3):
        """
        可视化特征图
        Args:
            features: 特征图字典
            save_dir: 保存目录（可选）
            max_channels: 每个特征图显示的最大通道数
        """
        for key, value in features.items():
            if key in ['image', 'image_id']:
                print(f"{key}: {value}")
            else:
                print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else [v.shape for v in value]}")


        image_id = features['image_id']
        num_features = len(features) - 2  # 减去 'image' 和 'image_id'

        # 动态调整子图布局
        cols = 4
        rows = math.ceil(num_features / cols) + 1  # 加1行用于显示原始图像
        plt.figure(figsize=(20, 5 * rows))

        # 显示原始图像（仅显示前3个通道）
        plt.subplot(rows, cols, 1)
        rgb_img = features['image'][:, :, :3]  # 取前三个通道作为RGB
        plt.imshow(rgb_img)
        plt.title(f"Original Image\n{rgb_img.shape}")
        plt.axis('off')

        # 显示各层特征图
        plot_idx = 2
        for name, feat in features.items():
            if name in ['image', 'image_id']:
                continue

            if isinstance(feat, list):
                for i, f in enumerate(feat):
                    if plot_idx > 16:
                        break
                    FeatureVisualizer._plot_feature(f, f"{name}[{i}]", plot_idx, max_channels)
                    plot_idx += 1
            else:
                if plot_idx > 16:
                    break
                FeatureVisualizer._plot_feature(feat, name, plot_idx, max_channels)
                plot_idx += 1

        plt.tight_layout()

        # 保存或显示
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_id}_features.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"特征图已保存至: {save_path}")
        else:
            plt.show()

    @staticmethod
    def _plot_feature(feat, title, position, max_channels):
        """绘制单个特征图"""
        plt.subplot(4, 4, position)

        # 转换为numpy数组
        if torch.is_tensor(feat):
            feat = feat.numpy()

        # 多通道特征图处理
        if len(feat.shape) == 3:
            # 显示前几个通道
            channels_to_show = min(feat.shape[0], max_channels)
            combined = np.zeros((feat.shape[1], feat.shape[2], channels_to_show))

            for c in range(channels_to_show):
                channel = feat[c]
                channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6)
                combined[:, :, c] = channel

             # 如果通道数为2，转换为伪彩色图像
            if combined.shape[2] == 2:
                pseudo_color = np.zeros((combined.shape[0], combined.shape[1], 3))
                pseudo_color[:, :, 0] = combined[:, :, 0]  # 第一个通道映射到红色
                pseudo_color[:, :, 1] = combined[:, :, 1]  # 第二个通道映射到绿色
                pseudo_color[:, :, 2] = 0  # 蓝色通道设置为零
                plt.imshow(pseudo_color)
            else:
                plt.imshow(combined)
        else:
            plt.imshow(feat, cmap='viridis')

        plt.title(f"{title}\n{feat.shape}")
        plt.axis('off')


# 使用示例
if __name__ == "__main__":
    # 1. 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFFNet().to(device)

    # 2. 创建可视化工具
    img_dir = r"D:\Project\PycharmProject\unet++\inputs\maweizao\train\image"  # 替换为TIFF图像目录
    visualizer = FeatureVisualizer(model, img_dir=img_dir, device=device)

    # 3. 选择要分析的图像ID（不带扩展名）
    img_id = "0_26_72"  # 替换为你的图像ID

    # 4. 获取并可视化特征图
    features = visualizer.get_feature_maps(img_id)
    visualizer.visualize(features, save_dir="./feature_maps/0_26_72")
