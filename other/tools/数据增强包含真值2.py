import os
from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A

# 配置输入和输出路径
input_image_dir = Path(r'C:\Users\zzc\Desktop\data\results\imgcut1')
input_gt_dir = Path(r'C:\Users\zzc\Desktop\data\results\cut01')
output_image_dir = Path(r'C:\Users\zzc\Desktop\data\Data enhancement\img\image')
output_gt_dir = Path(r'C:\Users\zzc\Desktop\data\Data enhancement\img\01')
# input_image_dir = Path(r'D:\jiaozhouwan\dataset\jiaozhouwan\CLCF\Data enhancement\1')
# input_gt_dir = Path(r'D:\jiaozhouwan\dataset\jiaozhouwan\CLCF\Data enhancement\class')
# output_image_dir = Path(r'D:\jiaozhouwan\dataset\jiaozhouwan\CLCF\Data enhancement\2')
# output_gt_dir = Path(r'D:\jiaozhouwan\dataset\jiaozhouwan\CLCF\Data enhancement\class2')

# 创建输出目录
output_image_dir.mkdir(parents=True, exist_ok=True)
output_gt_dir.mkdir(parents=True, exist_ok=True)

# 定义增强操作列表
augmentations = [
    A.HorizontalFlip(p=1.0),  #水平翻转
    A.VerticalFlip(p=1.0),    #垂直翻转
    A.RandomRotate90(p=1.0),  #随机旋转90
    A.Transpose(p=0.5)        #矩阵的转置
]

def load_image(filename):
    """Open image and convert image to array."""
    img = Image.open(filename)
    img = np.array(img).astype(np.uint8)
    return img

def save_image(image_array, filename):
    """Save numpy image array to a file."""
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)  # 将数据裁剪到0-255范围，并转换为uint8类型
    image = Image.fromarray(image_array)
    image.save(filename)

# 获取输入目录中的所有图像文件
image_files = list(input_image_dir.glob('*.tif'))

# 处理每一张图像及其对应的真值图
for img_file in image_files:
    gt_file = input_gt_dir / f"{img_file.stem}.tif"
    img = load_image(img_file)
    gt_image = load_image(gt_file)

    # 分别应用每个增强操作并保存
    for i, augmentation in enumerate(augmentations):
        augmented = augmentation(image=img, mask=gt_image)
        img_augmented = augmented['image']
        gt_augmented = augmented['mask']

        # 生成文件名编号
        base_filename = f"{img_file.stem}_{i}"
        file_index = 1500  # 开始编号
        output_filename = output_image_dir / f"{base_filename}_{file_index}.tif"
        output_gt_filename = output_gt_dir / f"{base_filename}_{file_index}.tif"

        # 确保编号唯一
        while output_filename.exists():
            file_index += 1
            output_filename = output_image_dir / f"{base_filename}_{file_index}.tif"
            output_gt_filename = output_gt_dir / f"{base_filename}_{file_index}.tif"

        # 保存图像和真值图
        save_image(img_augmented, output_filename)
        save_image(gt_augmented, output_gt_filename)

print(f'数据增强后的图片和真值图已分别保存到 {output_image_dir} 和 {output_gt_dir}')