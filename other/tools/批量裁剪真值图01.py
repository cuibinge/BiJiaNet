import os
import numpy as np
import imageio
from PIL import Image
Image.MAX_IMAGE_PIXELS = 200000000  # 设置为2亿像素
def cut_image_overlap(image, block_size, overlap_rate, save_path, start_block_num):
    # 检查重叠率是否在0到1之间
    if not (0 <= overlap_rate < 1):
        raise ValueError("重叠率必须在0到1之间。")

    # 获取图像尺寸
    row, col = image.shape
    dep = 1  # 二值图像的深度是1

    # 根据重叠率计算步长
    stride = int(block_size * (1 - overlap_rate))

    # 计算填充后的图像尺寸
    padding_h = ((row - block_size) // stride + 1) * stride + block_size
    padding_w = ((col - block_size) // stride + 1) * stride + block_size

    # 创建一个用于填充的图像
    padding_img = np.zeros((padding_h, padding_w), dtype=image.dtype)
    padding_img[:row, :col] = image[:row, :col]

    # 计算将要切割的块的数量
    row_blocks = (padding_h - block_size) // stride + 1
    col_blocks = (padding_w - block_size) // stride + 1

    # 初始化块编号
    block_num = start_block_num

    # 切割图像为块
    for i in range(row_blocks):
        for j in range(col_blocks):
            start_i = i * stride
            start_j = j * stride
            block = padding_img[start_i:start_i + block_size, start_j:start_j + block_size]
            block_name = f"{block_num}.tif"
            imageio.imwrite(os.path.join(save_path, block_name), block.astype(np.uint8) * 255)  # 确保保存为二值图像
            block_num += 1  # 更新块编号

    # 返回最后的块编号
    return block_num

# 主函数批量处理文件夹中的图像
if __name__ == "__main__":
    input_folder = r"C:\Users\zzc\Desktop\胶州湾影像\tif\12\shp"  # 输入标签文件夹路径
    output_folder = r"C:\Users\zzc\Desktop\胶州湾影像\tif\cut\shp"  # 输出标签文件夹路径
    block_size = 128 # 块大小
    overlap_rate = 0 # 重叠率

    # 确保输出路径存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有tif文件
    files = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.tif')]

    # 初始化块编号
    start_block_num = 0

    # 逐个处理文件
    for file in files:
        image_path = os.path.join(input_folder, file)
        image = imageio.v2.imread(image_path)
        # image = imageio.imread(image_path)

        # 如果图像是三维的，取第一个通道
        if image.ndim == 3:
            image = image[:,:,0]

        start_block_num = cut_image_overlap(image, block_size, overlap_rate, output_folder, start_block_num)