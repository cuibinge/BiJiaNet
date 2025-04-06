import os
import cv2
import time
import math
import json
import numpy as np
import shapely.geometry
from tqdm import tqdm

def preprocess_annotation(json_path, save_path, num_corners, is_training=True):
    """
    COCO格式多边形数据预处理脚本（针对PolyR-CNN优化）

    功能：
    1. 对训练集：将多边形顶点数填充至固定数量，并添加顶点类别标签
    2. 对测试集：仅进行数据清洗（去除噪声和无效多边形）
    """
    print(f"Loading annotation from {json_path}...")
    with open(json_path, "r") as f:
        labels = json.load(f)
    print("Annotation loaded.")

    annotations = labels["annotations"]
    indices = []  # 存储无效多边形的索引

    # 处理每个标注
    for i, anno in enumerate(tqdm(annotations, desc="Processing annotations")):
        # 获取多边形顶点
        gt_pts = anno["segmentation"][0]  # List [x1, y1, ..., xn, yn]

        # 计算并扩大边界框
        gt_bbox = get_gt_bboxes(gt_pts)  # [x1, y1, w, h]

        # 转换为numpy数组并去除冗余顶点
        gt_pts = np.array(gt_pts).reshape((-1, 2))  # numpy array, shape (N, 2), float64
        gt_pts = remove_doubles(gt_pts, epsilon=0.1)  # shape (N, 2), float64

        # 设置最小面积阈值（使训练集更宽松）
        min_area = 2 if is_training else 10

        # 过滤无效多边形（顶点数<3或面积过小）
        if gt_pts.shape[0] < 3 or shapely.geometry.Polygon(gt_pts).area < min_area:
            indices.append(i)
            continue

        # 多边形简化（减少顶点数）
        gt_pts = approximate_polygons(gt_pts, tolerance=0.01)  # shape (N, 2), float64

        # 再次检查有效性
        if gt_pts.shape[0] < 3 or shapely.geometry.Polygon(gt_pts).area < min_area:
            indices.append(i)
            continue

        # 从最高点开始重新排序顶点(多边形形状不会改变)
        gt_pts = np.array(gt_pts).reshape((-1, 2))
        ind = np.argmin(gt_pts[:, 1])
        gt_pts = np.concatenate((gt_pts[ind:], gt_pts[:ind]), axis=0)

        # 限制顶点坐标范围
        gt_pts = np.clip(gt_pts.flatten(), 0.0, 256 - 1e-4)  # shape, (N * 2), float64

        # 训练集特殊处理
        if is_training:
            # 顶点采样和分类标签生成
            gt_pts, gt_cor_cls = uniform_sampling(gt_pts, num_corners)  # (num_corners * 2), (num_corners,), int32
            annotations[i]["cor_cls_poly"] = [int(c) for c in gt_cor_cls]  # Add polygon corner classes
        # 测试集仅取整
        else:
            gt_pts = np.round(gt_pts).astype(np.int32)

        # 更新标注
        annotations[i]["segmentation"] = [[int(x) for x in gt_pts]]
        annotations[i]["bbox"] = gt_bbox

    # 移除无效标注（从后往前删除）
    indices = sorted(indices)
    for i in reversed(indices):
        annotations.pop(i)

    print(f"Processing complete. Saving to {save_path}...")

    # 保存处理后的文件
    labels["annotations"] = annotations
    with open(save_path, 'w') as f:
        json.dump(labels, f, indent=2)

    print("Annotation file saved.")

def uniform_sampling(gt_pts, num_corners):
    """
    将任意顶点数量的多边形统一采样为固定数量（num_corners）的顶点，同时区分原始顶点和插值点。（将原始顶点智能地融合到采样结果中，既保证了固定顶点数的要求，又最大限度地保留了多边形的几何特征）

    参数：
        polygon_points: 原始多边形顶点（一维数组）
        num_corners: 目标顶点数

    返回：
        调整后的顶点坐标和顶点类别标签（0=原始点，1=插值点）
    """
    # 初始化标签
    corner_label = np.ones((num_corners,), dtype=np.int32)

    # 创建多边形二值掩码
    polygon = np.round(gt_pts).astype(np.int32)
    polygon = polygon.reshape((-1, 1, 2))
    img = np.zeros((257, 257), dtype="uint8")
    img = cv2.polylines(img, [polygon], True, 255, 1)
    img = cv2.fillPoly(img, [polygon], 255)

    # 提取轮廓
    contour, _ = cv2.findContours(img, cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contour = contour[0]  # 取第一个轮廓
    lc = contour.shape[0]

    # 采样或填充顶点
    if lc >= num_corners:
        ind = np.linspace(start=0, stop=lc, num=num_corners, endpoint=False)
        ind = np.round(ind).astype(np.int32)
        encoded_polygon = contour[ind].reshape((-1, 2))
    else:
        contour = contour.reshape((-1, 2))
        contour = [list(x) for x in contour]
        encoded_polygon = contour + [contour[-1]] * (num_corners - lc)  # Pad with the last vertex
        encoded_polygon = np.array(encoded_polygon).reshape((-1, 2))

    # 标记原始顶点
    polygon = polygon.reshape((-1, 2))
    for i, x in enumerate(polygon):
        dists = [np.sqrt(np.sum((x - y) ** 2)) for y in encoded_polygon]  # 计算原始点与每个采样点之间的欧氏距离
        min_dist_idx = np.argmin(dists)  # 获得最近的点的索引
        # 距离为 0 ，说明当前原始顶点 x 已经存在于采样点 encoded_polygon 中，直接将该采样点的标签 corner_label[min_dist_idx] 设为 0
        if dists[min_dist_idx] == 0:
            corner_label[min_dist_idx] = 0
        # 距离不为 0 ，说明 x 未被采样到，将最近的采样点 encoded_polygon[min_dist_idx] 替换为 x，同时将该点标签设为 0
        else:
            encoded_polygon[min_dist_idx] = x
            corner_label[min_dist_idx] = 0

    return encoded_polygon.flatten(), corner_label

def get_gt_bboxes(gt_pts):
    """
    计算扩大20%的边界框
    """
    gt_pts = np.array(gt_pts).reshape((-1, 2))
    x = gt_pts[:, 0]
    y = gt_pts[:, 1]

    xmin = x.min()
    ymin = y.min()
    xmax = x.max()
    ymax = y.max()

    w, h = xmax - xmin, ymax - ymin

    # 扩大20%
    xmin = max(xmin - w * 0.1, 0.0)
    ymin = max(ymin - h * 0.1, 0.0)
    xmax = min(xmax + w * 0.1, 300 - 1e-4)
    ymax = min(ymax + h * 0.1, 300 - 1e-4)

    w = xmax - xmin
    h = ymax - ymin

    return [float(xmin), float(ymin), float(w), float(h)]


def remove_doubles(vertices, epsilon=0.1):
    """
    移除距离过近的冗余顶点.

    Args:
        vertices (numpy.ndarray): 形状为(N, 2)的多边形顶点
        epsilon (float, optional): Defaults to 0.1.
    """
    dists = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=-1)
    # np.roll(vertices, -1, axis=0): [[x_2, y_2], [x_3, y_3], ... , [x_n, y_n], [x_1, y_1]]
    # dists: [dist(v_2, v_1), dist(v_3, v_2), ... , dist(v_n, v_{n-1}), dist(v_1, v_n)]
    new_vertices = vertices[epsilon < dists]
    # If dist(v_{k}, v_{k-1}) < epsilon, remove v_{k-1}
    # If dist(v_{1}, v_{n}) < epsilon, remove v_{n}
    return new_vertices


def approximate_polygons(polygon, tolerance=0.01):
    """
    使用Douglas-Peucker算法简化多边形

    Args:
        polygon (numpy.ndarray): 形状为(N, 2)的多边形顶点
        tolerance (float, optional): Defaults to 0.01.
    """
    from skimage.measure import approximate_polygon
    return approximate_polygon(polygon, tolerance)

input_json = "../data/train/annotations.json"
output_json = "../data/train/annotation_preprocessed.json"
target_vertices = 64
preprocess_annotation(json_path=input_json, save_path=output_json, num_corners=target_vertices, is_training=True)