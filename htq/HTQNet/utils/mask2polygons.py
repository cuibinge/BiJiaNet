import os
import json
import imageio
import numpy as np
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage.measure import find_contours

def save_polygons_to_json(polygons, image_id, output_path):
    """
        将多边形数据保存为JSON文件
        :param polygons: 多边形顶点列表
        :param image_id: 图像文件名或唯一标识
        :param output_path: JSON文件保存路径
    """

    # 构建数据结构
    data = {
        "image_id": image_id,
        "polygons": polygons
    }

    # 保存为JSON文件
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)  # indent=4 用于美化输出

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

        # 将顶点坐标缩放到特征图分辨率(FPN)
        # exterior /= 4  # 假设 FPN 的输出分辨率是输入图像的 1/4

        # 过滤无效多边形
        if len(exterior) >= 3:

            # 计算边界框 (x_min, y_min, x_max, y_max)
            x_min, y_min = np.min(exterior, axis=0)
            x_max, y_max = np.max(exterior, axis=0)
            bbox = [float(x_min), float(y_min), float(x_max), float(y_max)]  # 转换为 Python float用于json文件序列化输出

            # 保存多边形和边界框
            polygons.append({
                "bbox": bbox,
                "polygon": exterior.tolist()  # 转换为列表格式
            })

    return polygons

def process_folder(image_folder, mask_folder, output_folder, json_output_folder):

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(json_output_folder):
        os.makedirs(json_output_folder)

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
        json_output_path = os.path.join(json_output_folder, f"{os.path.splitext(image_name)[0]}.json")

        # 将多边形叠加在图像上并保存
        plot_polygons_on_image(image, polygons, output_path)
        print(f"Processed and saved: {output_path}")

        # 保存多边形数据为JSON
        save_polygons_to_json(polygons, image_name, json_output_path)
        print(f"Saved polygons to: {json_output_path}")

def plot_polygons_on_image(image, polygons, output_path):

    # 创建一个画布
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # 显示原始图像
    ax.imshow(image)

    # 绘制提取的多边形
    for polygon_data in polygons:
        # 提取边界框
        bbox = polygon_data["bbox"]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        # 绘制边界框
        rect = patches.Rectangle(
            (x_min, y_min), width, height, linewidth=2, edgecolor='blue', facecolor='none'
        )
        ax.add_patch(rect)

        # 提取多边形顶点
        polygon = polygon_data["polygon"]
        polygon = np.array(polygon)  # 转换为 numpy 数组

        # 绘制多边形
        plt.plot(polygon[:, 0], polygon[:, 1], linewidth=2, color='red')

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

    image_dir = "../data/train/images"
    mask_dir = "../data/train/labels"
    output_dir = "../data/train/polygons"
    json_output_dir = "../data/train/jsons"
    process_folder(image_dir, mask_dir, output_dir, json_output_dir)