import os
import numpy as np
from PIL import Image

# 定义输入文件夹和输出文件夹
input_folder = 'C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\128\\shp'  # 替换为你的输入文件夹路径
output_folder = 'C:\\Users\\zzc\\Desktop\\胶州湾影像\\tif\\cut\\shp'  # 替换为你的输出文件夹路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.tif'):  # 检查文件是否是TIFF格式
        # 构造完整的文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开图像
        with Image.open(input_path) as img:
            # 转换为numpy数组
            img_array = np.array(img)

            # 归一化到0-1范围
            normalized_array = img_array.astype(np.float32) / 255.0

            # 将归一化后的数组转换回PIL图像
            normalized_img = Image.fromarray((normalized_array).astype(np.uint8))

            # 保存处理后的图像
            normalized_img.save(output_path)
            print(f"处理并保存了: {filename}")