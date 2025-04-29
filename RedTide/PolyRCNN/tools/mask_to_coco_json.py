import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from skimage import io
from collections import defaultdict
from shapely.geometry import Polygon


def visualize_polygons_with_matplotlib(image_dir, json_path, output_dir):
    """
    用Matplotlib绘制带有多边形和标注信息的图片（更清晰，适合检查细节）。

    Args:
        image_dir (str): 图片文件夹路径。
        json_path (str): annotations.json文件路径。
        output_dir (str): 输出目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    image_annos = defaultdict(list)
    for anno in coco_data['annotations']:
        image_annos[anno['image_id']].append(anno)

    for image_info in tqdm(coco_data['images'], desc="Visualizing"):
        image_id = image_info['id']
        image_path = os.path.join(image_dir, image_info['file_name'])
        output_path = os.path.join(output_dir, f"{image_info['file_name'][:-4]}.png")

        # 读取图片
        img = io.imread(image_path)

        # 创建画布
        fig, ax = plt.subplots()
        ax.imshow(img)

        # 绘制每个多边形
        for anno in image_annos.get(image_id, []):
            polygon = np.array(anno['segmentation'][0]).reshape(-1, 2)
            patch = patches.Polygon(
                polygon, closed=True,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            ax.add_patch(patch)

            # 可视化边界框
            # bbox = anno['bbox']
            # x, y, w, h = bbox
            # rect = patches.Rectangle(
            #     (x, y), w, h, linewidth=1, edgecolor='blue', facecolor='none'
            # )
            # ax.add_patch(rect)

            # 可选：添加顶点标记
            # ax.scatter(polygon[:, 0], polygon[:, 1], color='red', s=10)

        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()

def visualize_polygons_with_opencv(image_dir, json_path, output_dir):
    """
    用OpenCV将多边形绘制到图片上并保存。

    Args:
        image_dir (str): 图片文件夹路径（需与JSON中的file_name匹配）。
        json_path (str): annotations.json文件路径。
        output_dir (str): 可视化结果保存目录。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载COCO标注
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 按图片ID分组标注
    image_annos = defaultdict(list)
    for anno in coco_data['annotations']:
        image_annos[anno['image_id']].append(anno)

    # 处理每张图片
    for image_info in tqdm(coco_data['images'], desc="Visualizing"):
        image_id = image_info['id']
        image_path = os.path.join(image_dir, image_info['file_name'])
        output_path = os.path.join(output_dir, f"vis_{image_info['file_name']}.jpg")

        # 读取图片（四通道TIFF转三通道BGR）
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 4:  # 四通道转三通道（取RGB，丢弃近红外）
            img = img[:, :, 1:4]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 绘制当前图片的所有多边形
        for anno in image_annos.get(image_id, []):
            polygon = np.array(anno['segmentation'][0]).reshape(-1, 2).astype(np.int32)
            cv2.polylines(img, [polygon], isClosed=True, color=(0, 0, 255), thickness=1)
            bbox = anno['bbox']
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)

        # 保存结果
        cv2.imwrite(output_path, img)

def mask_to_coco_json(image_dir, mask_dir, output_json_path):
    """
    将三通道PNG图片和单通道PNG掩码转换为COCO格式的JSON标注文件。

    Args:
        image_dir (str): 存放训练图片的文件夹路径（三通道PNG格式）。
        mask_dir (str): 存放对应二值掩码的文件夹路径（单通道PNG格式）。
        output_json_path (str): 输出的JSON文件路径。
    """

    # 初始化COCO数据结构
    coco_data = {
        "info": {"description": "Converted from PNG masks"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "redTide", "supercategory": "none"}]
    }

    # 遍历图片文件夹
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    annotation_id = 1  # 标注ID从1开始

    for image_id, image_file in enumerate(tqdm(image_files, desc="Processing images")):
        # 1. 处理图片信息
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, image_file)  # 假设图片和掩码同名

        # 读取图片（仅获取尺寸，不需要实际数据）
        image = io.imread(image_path)
        height, width = image.shape[:2]

        # 添加到images字段
        coco_data["images"].append({
            "id": image_id + 1,  # COCO的image_id从1开始
            "file_name": image_file,
            "width": width,
            "height": height
        })

        # 2. 处理掩码（提取多边形）
        mask = io.imread(mask_path)
        if mask.ndim == 3:  # 确保是单通道
            mask = mask[:, :, 0]
        print(mask[mask != 0])
        mask = (mask > 0).astype(np.uint8) * 255  # 二值化
        print(mask[mask != 0])

        # 提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # 简化多边形（减少顶点数）
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 转换为COCO格式的segmentation（一维列表）
            if len(approx) >= 3:  # 至少需要3个顶点
                # 将顶点转换为列表并显式闭合
                points = approx.reshape(-1, 2).tolist()
                if points[0] != points[-1]:  # 如果首尾不重合，添加起点作为终点
                    points.append(points[0])
                segmentation = [coord for point in points for coord in point]  # 展平为一维列表

                # 计算面积和包围框
                polygon = Polygon(points)
                area = polygon.area
                x_min, y_min, w, h = cv2.boundingRect(np.array(points).reshape(-1, 1, 2))

                # 添加到annotations字段
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id + 1,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": float(area),
                    "bbox": [float(x_min), float(y_min), float(w), float(h)],
                    "iscrowd": 0
                })
                annotation_id += 1

    # 保存JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    print(f"COCO JSON saved to {output_json_path}")

if __name__ == '__main__':

    # 生成COCO格式的json标签
    # image_dir = "../data/train/images"
    # mask_dir = "../data/train/labels"
    # output_json_path = "../data/train/annotations.json"
    # mask_to_coco_json(image_dir, mask_dir, output_json_path)

    # 使用matplotlib可视化生成COCO格式的json标签
    image_dir = "../data/train/images"
    json_path = "../data/train/annotation_preprocessed.json"
    output_dir = "../data/train/polygons_preprocessed"
    visualize_polygons_with_matplotlib(image_dir, json_path, output_dir)

    # 使用opencv可视化生成COCO格式的json标签
    # image_dir = "../data/train/images"
    # json_path = "../data/train/annotations.json"
    # output_dir = "../data/train/polygons"
    # visualize_polygons_with_opencv(image_dir, json_path, output_dir)
