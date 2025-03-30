import os
import numpy as np
from PIL import Image

def calculate_mean_std(image_folder):
    """
    计算指定文件夹中所有图像的均值和标准差。
    """
    # 初始化变量
    total_mean = 0
    total_std = 0
    total_pixels = 0

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(image_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # 打开图像文件
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path).convert("RGB")  # 确保图像是 RGB 格式
            img_array = np.array(img, dtype=np.float32)

            # 计算当前图像的均值和标准差
            img_mean = np.mean(img_array, axis=(0, 1))  # 按通道计算均值
            img_std = np.std(img_array, axis=(0, 1))    # 按通道计算标准差

            # 累加统计值
            total_mean += img_mean * img_array.size / 3
            total_std += img_std * img_array.size / 3
            total_pixels += img_array.size / 3

    # 计算全局均值和标准差
    global_mean = total_mean / total_pixels
    global_std = total_std / total_pixels

    return global_mean, global_std

if __name__ == "__main__":
    # 指定图像文件夹路径
    image_folder = r"D:\Dataset\RedTide\train\masks"  # 替换为你的图像文件夹路径

    # 计算均值和标准差
    mean, std = calculate_mean_std(image_folder)

    # 打印结果
    print(f"Mean: {mean}")
    print(f"Std: {std}")