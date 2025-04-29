import os
import cv2
import numpy as np

def calculate_mean_and_std(folder_path):
    # 初始化用于存储所有图片各通道像素值的列表
    red_channel = []
    green_channel = []
    blue_channel = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片文件
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            # 构建图片文件的完整路径
            file_path = os.path.join(folder_path, filename)
            # 使用OpenCV读取图片
            img = cv2.imread(file_path)
            # 将图片从BGR格式转换为RGB格式
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 分离RGB通道
            r, g, b = cv2.split(img)

            # 将每个通道的像素值添加到对应的列表中
            red_channel.extend(r.flatten())
            green_channel.extend(g.flatten())
            blue_channel.extend(b.flatten())

    # 将列表转换为NumPy数组
    red_channel = np.array(red_channel)
    green_channel = np.array(green_channel)
    blue_channel = np.array(blue_channel)

    # 计算每个通道的均值和标准差
    mean_red = np.mean(red_channel)
    std_red = np.std(red_channel)
    mean_green = np.mean(green_channel)
    std_green = np.std(green_channel)
    mean_blue = np.mean(blue_channel)
    std_blue = np.std(blue_channel)

    return (mean_red, mean_green, mean_blue), (std_red, std_green, std_blue)

# 指定包含RGB图片的文件夹路径
folder_path = '../data/train/images'
# 调用函数计算均值和标准差
means, stds = calculate_mean_and_std(folder_path)

print(f"Mean values (R, G, B): {means}")
print(f"Standard deviation values (R, G, B): {stds}")