import os
import imageio
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage.measure import find_contours

def mask_to_polygons(mask, threshold=0):
    """
        将二值掩码转换为多边形顶点序列
        :param mask: 二值掩码，形状为 (H, W)
        :param threshold: 二值化阈值
        :param image_size: 图像的尺寸 (H, W)
        :return: 多边形顶点列表，每个元素是一个多边形的顶点序列 (N, 2)
    """
    # 二值化标签
    binary_mask = mask > threshold
    polygons = []

    # 提取轮廓
    contours = find_contours(binary_mask, 0.5)
    for contour in contours:

        # 简化多边形（减少顶点数量）
        polygon = Polygon(contour).simplify(1.0)
        if polygon.is_empty:
            continue

        # 获取外环顶点坐标，并交换 x 和 y 坐标
        exterior = np.array(polygon.exterior.coords).astype(np.float32)
        exterior[:, [0, 1]] = exterior[:, [1, 0]]  # 交换 x 和 y 坐标
        polygons.append(exterior)

        # 将顶点坐标缩放到特征图分辨率(FPN)
        exterior /= 4  # 假设 FPN 的输出分辨率是输入图像的 1/4

    return polygons

def process_folder(image_folder, mask_folder, output_folder):

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历图像文件夹中的所有文件
    for image_name in os.listdir(image_folder):
        # 构建图像和掩码的路径
        image_path = os.path.join(image_folder, image_name)
        mask_path = os.path.join(mask_folder, image_name)  # 假设图像和掩码文件名相同

        # 检查掩码文件是否存在
        if not os.path.exists(mask_path):
            print(f"Mask file for {image_name} not found, skipping.")
            continue

        # 读取图像和掩码
        image = imageio.imread(image_path)
        mask = imageio.imread(mask_path)

        # 提取多边形
        polygons = mask_to_polygons(mask)

        # 构建输出路径
        output_path = os.path.join(output_folder, image_name)

        # 将多边形叠加在图像上并保存
        plot_polygons_on_image(image, polygons, output_path)
        print(f"Processed and saved: {output_path}")

def plot_polygons_on_image(image, polygons, output_path):
    """
    将多边形叠加在图像上并保存
    :param image: 输入图像
    :param polygons: 多边形顶点列表
    :param output_path: 输出路径
    """
    # 创建一个画布
    plt.figure(figsize=(10, 10))

    # 显示原始图像
    plt.imshow(image)

    # 绘制提取的多边形
    for polygon in polygons:
        # 将顶点坐标放大到输入图像的分辨率
        polygon *= 4  # 假设输入图像的分辨率是特征图的 4 倍

        # 基于主对角线对折多边形的顶点
        flipped_polygon = np.zeros_like(polygon)
        for i, (x, y) in enumerate(polygon):
            flipped_polygon[i] = [y, x]  # 交换 x 和 y 坐标

        # 绘制对折后的多边形
        #plt.plot(flipped_polygon[:, 1], flipped_polygon[:, 0], linewidth=2, color='red')  # 注意坐标顺序

        plt.plot(flipped_polygon[:-1, 1], flipped_polygon[:-1, 0], linewidth=2, color='red')
        # 检查首尾点是否一致
        # if not np.array_equal(flipped_polygon[0], flipped_polygon[-1]):
        #     # 如果首尾点不一致，只绘制除首尾连接之外的部分
        #     plt.plot(flipped_polygon[:-1, 1], flipped_polygon[:-1, 0], linewidth=2, color='red')
        # else:
        #     # 如果首尾点一致，绘制整个多边形
        #     plt.plot(flipped_polygon[:, 1], flipped_polygon[:, 0], linewidth=2, color='red')
    # 关闭坐标轴
    plt.axis('off')

    # 保存结果图像
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
if __name__ == '__main__':
    # mask_path = "../data/train/labels/0_1_3.tif"
    # mask = imageio.imread(mask_path)
    # polygons = mask_to_polygons(mask)
    # print(polygons)
    # print("提取的多边形数量:", len(polygons))
    # for i, polygon in enumerate(polygons):
    #     print(f"多边形 {i} 的顶点数:", len(polygon))

    image_dir = "inputs/PNGData/test/img"
    mask_dir = "outputs/PNGData_SFFNet_woDS/0"
    output_dir = "outputs/PNGData_SFFNet_woDS/0/222"
    process_folder(image_dir, mask_dir, output_dir)