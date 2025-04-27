import argparse
import glob
import os
import cv2
import tqdm
import numpy as np
import torch
import shapely.geometry

from shapely.geometry import Polygon, LineString
from shapely.validation import make_valid
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from tqdm import tqdm

def get_parser():
    """
    Define the argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Model inference and visualization script")
    parser.add_argument("--config-file", required=True, help="Path to the model configuration file (YAML).")
    parser.add_argument("--input", nargs="+", help="Input images. Accepts one or more image paths separated by spaces, "
             "or a glob pattern like 'path/to/folder/*.jpg' to process multiple images in a directory.")
    # The '--input' argument accepts input images in several formats:
    #
    # 1. Single image path:
    #    Example:
    #        python demo.py --input path/to/image.jpg --output path/to/output/
    #    In this case, 'args.input' will be a list with one element:
    #        args.input = ["path/to/image.jpg"]
    #
    # 2. Multiple image paths:
    #    Example:
    #        python demo.py --input path/to/image1.jpg path/to/image2.jpg --output path/to/output/
    #    Here, 'args.input' will be a list of the specified image paths:
    #        args.input = ["path/to/image1.jpg", "path/to/image2.jpg"]
    #
    # 3. Glob pattern matching multiple images:
    #    Example:
    #        python script.py --input "path/to/images/*.jpg" --output path/to/output/
    #    The script will expand the glob pattern to all matching files:
    #        args.input = ["path/to/images/image1.jpg", "path/to/images/image2.jpg", ...]
    #
    # In the code, if 'args.input' contains only one element (which could be a glob pattern),
    # it is expanded using 'glob.glob' to find all matching files:
    #
    #     if len(args.input) == 1:
    #         args.input = glob.glob(os.path.expanduser(args.input[0]))
    #         assert args.input, "No files found for the provided input pattern."
    #
    # This ensures that 'args.input' is always a list of file paths, making it flexible to accept
    # single images, multiple images, or patterns matching multiple images.
    parser.add_argument("--output", required=True, help="Directory to save the output visualizations. Will create the directory if it doesn't exist.")
    parser.add_argument("--confidence-threshold", type=float, default=0.1, help="Confidence threshold for predictions")  # 0.5
    parser.add_argument("--corner-threshold", type=float, default=0, help="Threshold for vertex corner classification")  # 0.2
    parser.add_argument("--nms", action="store_true", help="Whether to apply NMS to polygon vertices.")
    parser.add_argument("--merge", action="store_true", help="Whether to merge edges of polygons with angles smaller than 10 degrees.")
    parser.add_argument("--nms_bbox", action="store_true", help="Whether to apply NMS to building instances (bounding boxes).")
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    return parser


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    from polyrcnn import add_polyrcnn_config
    add_polyrcnn_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def visualize_polygons(img, polygons):
    """
    Visualize polygons on an image with distinct colors for each polygon.

    :param img: ndarray, the original image.
    :param polygons: list of list, each inner list represents a polygon as
                     a sequence of vertices in the format:
                     [[x1, y1, x2, y2, ...], [x1, y1, x2, y2, ...], ...]
    :return: ndarray, the image with visualized polygons and vertices.
    """
    colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0),
              (0, 255, 155), (0, 0, 255), (160, 32, 240)]
    for i, polygon in enumerate(polygons):
        polygon = np.array(polygon).flatten()
        polygon = np.round(polygon).astype(np.int32).reshape((-1, 1, 2)) * 3  # enlarge polygon for visualization
        col = colors[i % len(colors)]
        img = cv2.polylines(img, [polygon], True, col, 2)
        # polygon = polygon.reshape((-1, 2))
        # for cor in polygon:
        #     img = cv2.circle(img, (cor[0], cor[1]), 4, col, -1)
    return img


def visualize_bboxes(img, bboxes):
    """
    Visualize bounding boxes on an image with distinct colors for each box.

    :param img: ndarray, the original image.
    :param bboxes: list of list, each inner list represents a bounding box in the format:
                   [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
                   where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    :return: ndarray, the image with visualized bounding boxes.
    """
    colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
              (255, 255, 0), (255, 0, 0), (240, 32, 160)]
    for i, bbox in enumerate(bboxes):
        bbox = np.round(np.array(bbox)).astype(np.int32) * 3  # enlarge bbox for visualization
        col = colors[i % len(colors)]
        img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), col, 2)
    return img


def fix_self_intersections(polygon):
    """
    Fix self-intersecting polygons by reordering vertices and creating valid geometry.

    :param polygon: np.ndarray, shape (N, 2), the coordinates of polygon vertices
    :return: np.ndarray, the fixed polygon vertices
    """
    # 转换为Shapely polygon
    shapely_poly = Polygon(polygon)

    # 检查多边形是不是合理的
    if not shapely_poly.is_valid:
        # Fix self-intersections by making the polygon valid
        print("-------------------------------------")
        valid_poly = make_valid(shapely_poly)

        # Handle different output types from make_valid
        if isinstance(valid_poly, Polygon):
            # Simple case - got back a single polygon
            fixed_coords = np.array(valid_poly.exterior.coords)
        elif isinstance(valid_poly, MultiPolygon):
            # Complex case - got multiple polygons, take the largest one
            largest_poly = max(valid_poly.geoms, key=lambda p: p.area)
            fixed_coords = np.array(largest_poly.exterior.coords)
        elif isinstance(valid_poly, GeometryCollection):
            # Extract polygons from geometry collection
            polygons = [g for g in valid_poly.geoms if isinstance(g, Polygon)]
            if polygons:
                largest_poly = max(polygons, key=lambda p: p.area)
                fixed_coords = np.array(largest_poly.exterior.coords)
            else:
                # Fallback to original if no valid polygons found
                fixed_coords = polygon
        else:
            # Fallback to original if unexpected type
            fixed_coords = polygon

        return fixed_coords

    # Return original if already valid
    return polygon


def sort_polygon_vertices(polygon):
    """
    Sort polygon vertices in clockwise order to potentially fix self-intersections.

    :param polygon: np.ndarray, shape (N, 2), the coordinates of polygon vertices
    :return: np.ndarray, the sorted polygon vertices
    """
    # Calculate centroid
    centroid = np.mean(polygon, axis=0)

    # Calculate angles from centroid to each point
    vectors = polygon - centroid
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Sort vertices by angle
    sorted_indices = np.argsort(angles)

    # Return sorted vertices (clockwise order)
    return polygon[sorted_indices]

def reduce_redundant_vertices(corner_scores, polygons, corner_threshold, bbox_scores, bboxes, nms, merge, nms_thres=2, merge_thres=10):
    """
    对一张图像中预测到的每个多边形减少冗余的顶点

    :param corner_scores: torch.Size([N, M]), 每张图像中预测到的多边形的每个顶点是角点的概率
    :param polygons: torch.Size([N, M, 2]), 每张图像中预测到的多边形的顶点
    :param corner_threshold: float, corner threshold
    :param bbox_scores: torch.Size([N]), 每张图像中预测到的多边形的概率
    :param bboxes: torch.Size([N, 4]), 每张图像中预测到的多边形的坐标
    :param nms: bool, whether to apply NMS to reduce redundant corners
    :param merge: bool, whether to merge edges with angles smaller than a threshold
    :param nms_thres: float, threshold for NMS on corners
    :param merge_thres: float, threshold for merging edges based on angles

    :return pred_polygons_thres: List[np.ndarray], processed polygons
    :return pred_corners_scores: List[np.ndarray], scores of the corners for the processed polygons
    :return pred_bboxes: List[np.ndarray], bounding boxes
    :return pred_bboxes_scores: List[np.ndarray], scores of the bounding boxes
    """
    pred_polygons_thres = []
    pred_corners_scores = []

    pred_bboxes = []
    pred_bboxes_scores = []

    for i in range(len(polygons)):
        pred_bbox = bboxes[i]
        pred_bbox_score = bbox_scores[i]

        # 基于corner scores进行初始过滤
        mask = (corner_scores[i] > corner_threshold).nonzero()[0].flatten()
        pred_polygon = polygons[i][mask]  # torch.Size(n, 2)
        pred_corner_scores = corner_scores[i][mask]

        # 总是移除定点数少于3的多边形或者多边形像素数量少于10的多边形
        # if pred_polygon.shape[0] <= 2 or shapely.geometry.Polygon(pred_polygon).area <= 10:
        #     continue
        if pred_polygon.shape[0] <= 2:
            continue

        pred_polygon = np.array(pred_polygon)
        pred_corner_scores = np.array(pred_corner_scores)
        pred_bbox = np.array(pred_bbox)
        pred_bbox_score = np.array(pred_bbox_score)

        # Optional: NMS on corners
        if nms:
            keep = nms_corner(pred_polygon, pred_corner_scores, nms_thres)
            pred_polygon = pred_polygon[keep]
            pred_corner_scores = pred_corner_scores[keep]

        # Optional: Merge edges with small angles
        if merge:
            while True:
                if pred_polygon.shape[0] <= 2:
                    break
                simplified_pred_polygon = simple_polygon(pred_polygon, merge_thres)
                if simplified_pred_polygon.shape[0] == pred_polygon.shape[0]:
                    break
                pred_polygon = simplified_pred_polygon

        # Revalidate polygons after optional operations
        # if pred_polygon.shape[0] <= 2 or shapely.geometry.Polygon(pred_polygon).area <= 10:
        #     continue

        if pred_polygon.shape[0] <= 2:
            continue

        pred_polygons_thres.append(pred_polygon)
        pred_corners_scores.append(pred_corner_scores)
        pred_bboxes.append(pred_bbox)
        pred_bboxes_scores.append(pred_bbox_score)

    return pred_polygons_thres, pred_corners_scores, pred_bboxes, pred_bboxes_scores


def nms_corner(poly, scores, thres):
    """
    Perform Non-Maximum Suppression (NMS) on polygon vertices to remove redundant vertices.

    :param poly: np.ndarray, shape (N, 2), the coordinates of polygon vertices.
                 N is the number of vertices.
    :param scores: np.ndarray, shape (N,), the classification scores of each vertex.
    :param thres: float, the distance threshold for suppressing vertices.
    :return: np.ndarray, indices of the vertices to keep after NMS.
    """
    # Sort vertices by their scores in descending order
    index = np.argsort(scores)[::-1]  # Indices of vertices sorted by score
    keep = []  # Indices of vertices to keep

    while index.shape[0] > 0:
        # Select the vertex with the highest score
        i = index[0]
        keep.append(i)

        # Calculate distances between the selected vertex and the remaining vertices
        dx = poly[i, 0] - poly[index[1:], 0]
        dy = poly[i, 1] - poly[index[1:], 1]
        d = np.sqrt(dx ** 2 + dy ** 2)  # Euclidean distance

        # Keep vertices that are farther than the threshold
        idx = np.where(d > thres)[0]

        # Update index to exclude suppressed vertices
        index = index[idx + 1]

    # Sort the kept indices for consistent ordering
    keep = np.sort(keep)

    return keep


def nms_bbox(bboxes, bbox_scores, thres):
    """
    Perform Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.

    Args:
        bboxes (np.ndarray): Array of bounding boxes with shape (N, 4), where N is the number of boxes.
                             Each box is represented as [x1, y1, x2, y2].
        bbox_scores (np.ndarray): Array of scores for each bounding box with shape (N,).
        thres (float): IoU threshold. Boxes with IoU >= thres will be suppressed.

    Returns:
        np.ndarray: Indices of the bounding boxes to keep.
    """
    # Convert inputs to NumPy arrays for consistent processing
    bboxes = np.array(bboxes)
    bbox_scores = np.array(bbox_scores)

    # Extract coordinates of each bounding box
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Compute area of each bounding box
    areas = (y2 - y1) * (x2 - x1)

    # Initialize list of indices to keep
    keep = []

    # Sort indices by scores in descending order
    index = bbox_scores.argsort()[::-1]

    # Perform NMS
    while index.shape[0] > 0:
        # Index of the current highest scoring box
        i = index[0]
        keep.append(i)

        # Compute IoU with remaining boxes
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        # Compute width and height of the intersection
        w = np.maximum(x22 - x11, 0)
        h = np.maximum(y22 - y11, 0)

        # Compute area of the intersection
        overlaps = w * h

        # Compute IoU
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

        # Filter out boxes with IoU >= threshold
        idx = np.where(ious < thres)[0]

        # Update indices (skip suppressed boxes)
        index = index[idx + 1]

    # Return sorted indices of kept boxes
    return np.sort(keep)


def simple_polygon(poly, thres):
    """
    Simplifies a polygon by merging edges with angles smaller than a threshold.

    :param poly: np.ndarray, shape (N, 2), vertices of the polygon
                 N is the number of vertices in the polygon.
    :param thres: float, angle threshold in degrees for merging edges.
    :return: np.ndarray, simplified polygon with reduced vertices.
    """
    # Remove the duplicate start and end points if the polygon is closed
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]

    # Compute edge vectors
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)  # Create edge segments
    vec0 = lines[:, 2:] - lines[:, :2]  # Current edge vectors
    vec1 = np.roll(vec0, -1, axis=0)  # Next edge vectors

    # Calculate angles of the edge vectors
    vec0_ang = np.arctan2(vec0[:, 1], vec0[:, 0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:, 1], vec1[:, 0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)  # Angle differences between consecutive edges

    # Flags to retain edges that do not meet the merging criteria
    flag1 = np.roll((lines_ang > thres), 1, axis=0)  # Larger than threshold
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)  # Smaller than complementary threshold

    # Keep only vertices that satisfy both angle criteria
    simple_poly = poly[np.bitwise_and(flag1, flag2)]

    return simple_poly


def main():
    args = get_parser().parse_args()

    # 添加配置和模型
    cfg = setup_cfg(args)
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # If the input contains a single glob pattern (e.g., "*.jpg"),
    # expand it into a list of matching file paths.
    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "No files found for the provided input pattern."

    # 确保输出路径存在
    os.makedirs(args.output, exist_ok=True)

    # 遍历每张图片
    for path in tqdm(args.input, desc="Processing images"):
        # 读取并预处理图片
        image = read_image(path, format="RGB")
        image_height, image_width = image.shape[:2]

        # 进行推理
        with torch.no_grad():
            predictions = model([{
                "image": torch.as_tensor(image.transpose(2, 0, 1).astype("float32")),
                "height": image_height,
                "width": image_width
            }])[0]

        # 打印推理结果调试
        # print(predictions)

        # 根据阈值提取预测结果
        instances = predictions["instances"].to("cpu")
        instances = instances[instances.scores > args.confidence_threshold]
        # print(instances)
        corner_scores = instances.corner_scores.numpy()  # (BS, NUM_CORNERS)
        N, M = corner_scores.shape
        contours = instances.pred_polygons.numpy().reshape(-1, M, 2)  # (BS, NUM_CORNERS, 2)
        bboxes = instances.pred_boxes.tensor.numpy()  # (BS, 4)
        bbox_scores = instances.scores.numpy()  # (BS,)

        if args.nms_bbox:
            keep = nms_bbox(bboxes, bbox_scores, thres=0.5)
            corner_scores = corner_scores[keep]
            contours = contours[keep]
            bboxes = bboxes[keep]
            bbox_scores = bbox_scores[keep]

        # 后处理预测到的多边形
        pred_polygons_thres, pred_corners_scores, pred_bboxes, pred_bboxes_scores = reduce_redundant_vertices(
            corner_scores, contours, args.corner_threshold, bbox_scores, bboxes,
            nms=args.nms, merge=args.merge, nms_thres=2, merge_thres=10
        )

        # 扩大图像的尺寸用于可视化
        enlarged_image = cv2.resize(image, (image_width * 3, image_height * 3))

        # 可视化边界框和多边形
        image_with_bboxes = visualize_bboxes(enlarged_image.copy(), pred_bboxes)
        image_with_polygons = visualize_polygons(enlarged_image.copy(), pred_polygons_thres)

        # 保存结果
        base_filename = os.path.splitext(os.path.basename(path))[0]
        # cv2.imwrite(os.path.join(args.output, f"{base_filename}_bboxes.jpg"), image_with_bboxes[:, :, ::-1])
        cv2.imwrite(os.path.join(args.output, f"{base_filename}_polygons.jpg"), image_with_polygons[:, :, ::-1])


if __name__ == "__main__":
    main()
