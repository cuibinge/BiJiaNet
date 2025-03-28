import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import os
import cv2
import numpy as np
import random

def get_visual_extreme_points(pts, img_width, img_height, tolerance=5):
    """
    获取视觉上的特征边界点（解决边缘裁剪导致的异常情况）
    
    返回：按顺时针排序的4-8个关键点（自动去重）
    """
    # 1. 初步获取传统极值点
    traditional = [
        pts[np.argmin(pts[:, 0])],  # x_min
        pts[np.argmax(pts[:, 1])],  # y_max
        pts[np.argmax(pts[:, 0])],  # x_max
        pts[np.argmin(pts[:, 1])]   # y_min
    ]
    
    # 2. 动态检测边缘接触点
    edge_points = []
    for pt in pts:
        x, y = pt
        # 判断是否接触图像边界（带容差）
        if (abs(x) <= tolerance or abs(x - img_width) <= tolerance or
            abs(y) <= tolerance or abs(y - img_height) <= tolerance):
            edge_points.append(pt)
    
    # 3. 混合策略提取关键点
    key_points = []
    
    # 上边缘：取最左和最右的点（y≈y_max）
    upper_mask = pts[:, 1] >= (traditional[1][1] - tolerance)
    if np.any(upper_mask):
        upper_pts = pts[upper_mask]
        key_points.extend([upper_pts[np.argmin(upper_pts[:, 0])],
                         upper_pts[np.argmax(upper_pts[:, 0])]])
    
    # 下边缘：取最左和最右的点（y≈y_min）
    lower_mask = pts[:, 1] <= (traditional[3][1] + tolerance)
    if np.any(lower_mask):
        lower_pts = pts[lower_mask]
        key_points.extend([lower_pts[np.argmin(lower_pts[:, 0])],
                         lower_pts[np.argmax(lower_pts[:, 0])]])
    
    # 左边缘：取最上和最下的点（x≈x_min）
    left_mask = pts[:, 0] <= (traditional[0][0] + tolerance)
    if np.any(left_mask):
        left_pts = pts[left_mask]
        key_points.extend([left_pts[np.argmax(left_pts[:, 1])],
                         left_pts[np.argmin(left_pts[:, 1])]])
    
    # 右边缘：取最上和最下的点（x≈x_max）
    right_mask = pts[:, 0] >= (traditional[2][0] - tolerance)
    if np.any(right_mask):
        right_pts = pts[right_mask]
        key_points.extend([right_pts[np.argmax(right_pts[:, 1])],
                         right_pts[np.argmin(right_pts[:, 1])]])
    
    # 4. 合并结果并去重
    # all_points = np.vstack([traditional, edge_points, key_points])
    # unique_points = np.unique(all_points, axis=0)
    # 合并时检查维度
    if len(edge_points) > 0:
        edge_points = np.array(edge_points)
        if edge_points.ndim == 1:
            edge_points = edge_points.reshape(-1, 2)  # 确保是 (M, 2)
        all_points = np.vstack([traditional, edge_points, key_points])
        # all_points = np.vstack([edge_points, key_points])
    else:
        all_points = traditional

    # return np.unique(all_points, axis=0)
    unique_points = np.unique(all_points, axis=0)
    # 5. 按顺时针排序（确保多边形不交叉）
    if len(unique_points) >= 3:
        center = np.mean(unique_points, axis=0)
        angles = np.arctan2(unique_points[:,1]-center[1], unique_points[:,0]-center[0])
        return unique_points[np.argsort(angles)]
    else:
        return unique_points
    
    
def visualize_contour_points(img, contours, contour_idx=0, show_indices=True):
    """
    生成轮廓点集可视化图（高精度版）
    
    :param img: 原始图像
    :param contours: 轮廓列表
    :param contour_idx: 轮廓索引
    :param show_indices: 是否显示点索引
    :return: 可视化图像
    """
    # 创建画布
    if len(img.shape) == 2:
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        debug_img = img.copy()
    
    # 获取指定轮廓
    cnt = contours[contour_idx]
    pts = np.squeeze(cnt)
    
    # 绘制所有点（带序号标记）
    for i, pt in enumerate(pts):
        # 绘制点（蓝色）
        cv2.circle(debug_img, tuple(pt.astype(int)), 2, (255, 0, 0), -1)
        
        # 绘制连线（绿色）
        if i < len(pts)-1:
            next_pt = pts[i+1]
            cv2.line(debug_img, tuple(pt.astype(int)), tuple(next_pt.astype(int)), 
                    (0, 255, 0), 1)
        
        # 显示点索引（白色）
        if show_indices:
            cv2.putText(debug_img, str(i), tuple(pt.astype(int) + np.array([5,-5])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    
    # 标记特殊点
    cv2.circle(debug_img, tuple(pts[0].astype(int)), 7, (0, 0, 255), -1)  # 红色起点
    cv2.circle(debug_img, tuple(pts[-1].astype(int)), 7, (0, 255, 0), -1) # 绿色终点
    
    # 添加信息标注
    info = [
        f"Contour {contour_idx}",
        f"Total points: {len(pts)}",
        "Red: Start | Green: End | Blue: Points"
    ]
    
    for i, text in enumerate(info):
        cv2.putText(debug_img, text, (10, 20+i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    return debug_img
def print_contour_points(contours, contour_idx=0):
    """
    打印指定轮廓的所有点坐标
    
    :param contours: 轮廓列表
    :param contour_idx: 轮廓索引
    """
    if contour_idx >= len(contours):
        print(f"错误：轮廓索引 {contour_idx} 超出范围（最大 {len(contours)-1}）")
        return
    
    cnt = contours[contour_idx]
    pts = np.squeeze(cnt)
    
    print(f"\n=== 轮廓 {contour_idx} 的坐标点 ===")
    print(f"总点数: {len(pts)}")
    print("所有坐标点 (x,y):")
    for i, pt in enumerate(pts):
        print(f"点{i}: ({pt[0]:.1f}, {pt[1]:.1f})")
    
    # 检查闭合状态
    if len(pts) > 1:
        distance = np.linalg.norm(pts[0] - pts[-1])
        print(f"\n闭合状态: {'是' if distance < 1.0 else '否'} (首尾距离: {distance:.2f}像素)")
def sort_points_clockwise(points):
    """将点按顺时针顺序排序"""
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:,1]-center[1], points[:,0]-center[0])
    return points[np.argsort(angles)]
def get_extreme_points(pts, img_width, img_height):
    if len(pts) == 0:
        return np.empty((0, 2))  # 返回空数组，避免后续错误

    # 提取极值点
    extremes = np.array([
        pts[np.argmin(pts[:, 0])],
        pts[np.argmax(pts[:, 1])],
        pts[np.argmax(pts[:, 0])],
        pts[np.argmin(pts[:, 1])]
    ])

    # 检查边缘点
    edge_points = []
    tolerance = 1
    for pt in pts:
        x, y = pt
        if (abs(x) <= tolerance or abs(x - img_width) <= tolerance or
            abs(y) <= tolerance or abs(y - img_height) <= tolerance):
            edge_points.append(pt)

    # 合并时检查维度
    if len(edge_points) > 0:
        edge_points = np.array(edge_points)
        if edge_points.ndim == 1:
            edge_points = edge_points.reshape(-1, 2)  # 确保是 (M, 2)
        all_points = np.vstack([extremes, edge_points])
    else:
        all_points = extremes

    return np.unique(all_points, axis=0)
def visualize_contours(img, contours, highlight_extremes=False, show_indices=False):
    """
    改进版轮廓可视化函数 - 显示所有轮廓的所有点
    
    :param img: 原始图像（单通道或三通道）
    :param contours: 轮廓列表
    :param highlight_extremes: 是否高亮显示四个极值点
    :param show_indices: 是否显示点的索引编号
    :return: 可视化图像
    """
    # 创建彩色调试图像
    if len(img.shape) == 2:
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        debug_img = img.copy()
    
    # 生成随机颜色（固定种子保证可重复）
    random.seed(42)
    colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
              for _ in range(len(contours))]
    
    for i, cnt in enumerate(contours):
        pts = np.squeeze(cnt)
        color = colors[i]
        
        # 绘制所有点（带连线）
        for j, pt in enumerate(pts):
            # 绘制点
            cv2.circle(debug_img, tuple(pt.astype(int)), 4, color, -1)
            
            # 绘制连线（除最后一个点）
            if j < len(pts)-1:
                next_pt = pts[j+1]
                cv2.line(debug_img, tuple(pt.astype(int)), tuple(next_pt.astype(int)), color, 1)
            
            # 显示点索引
            if show_indices:
                cv2.putText(debug_img, str(j), tuple(pt.astype(int)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        # 标记首尾点（红色起点，绿色终点）
        cv2.circle(debug_img, tuple(pts[0].astype(int)), 6, (0,0,255), -1)
        cv2.circle(debug_img, tuple(pts[-1].astype(int)), 6, (0,255,0), -1)
        
        # 高亮极值点（如果启用）
        if highlight_extremes and len(pts) >= 4:
            extremes = [
                pts[np.argmin(pts[:, 0])],  # x_min
                pts[np.argmax(pts[:, 1])],  # y_max
                pts[np.argmax(pts[:, 0])],  # x_max
                pts[np.argmin(pts[:, 1])]   # y_min
            ]
            for ext in extremes:
                cv2.circle(debug_img, tuple(ext.astype(int)), 8, (255,255,0), 2)  # 黄色高亮
        
        # 添加轮廓信息标注
        text = f"C{i}: {len(pts)}pts"
        cv2.putText(debug_img, text, tuple(pts[0].astype(int)-np.array([0,15])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # 添加图例说明
    legend = [
        "Red: Start point",
        "Green: End point",
        f"Total contours: {len(contours)}"
    ]
    if highlight_extremes:
        legend.append("Yellow: Extreme points")
    
    for i, text in enumerate(legend):
        cv2.putText(debug_img, text, (10, 20+i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    return debug_img
def transform_coords(pts, img_height):
    """强制调试版转换函数"""
    print("=== 转换前 ===")
    print("类型:", type(pts), "形状:", pts.shape)
    print("示例点:", pts[0])
    
    pts = pts.copy().astype(float)
    pts[:, 1] = img_height - pts[:, 1] - 1
    
    print("=== 转换后 ===")
    print("示例点:", pts[0])
    return pts
def jpg_to_shapefile(input_jpg, output_shp, output_contours, threshold=128, simplify_tolerance=0.5, min_area=100):
    try:
        # 1. 读取图像
        if not os.path.exists(input_jpg):
            raise FileNotFoundError(f"输入文件不存在: {input_jpg}")
            
        img = cv2.imread(input_jpg, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        
        if img is None:
            raise ValueError("无法读取图像，请检查文件格式")
        
        def transform_coords(pts, img_height=None):
            if img_height is not None:
                # Y轴翻转（OpenCV坐标系 → 笛卡尔坐标系）
                pts = pts.copy()  # 避免修改原数组
                pts[:, 1] = img_height - pts[:, 1] - 1  # 关键公式
    
            # 其他转换（如缩放/平移）可在此添加
            return pts
        
        # 2. 图像预处理
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # 3. 改进的边缘检测
        kernel = np.ones((3,3), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        edges = cv2.Canny(binary, 50, 150)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # 椭圆形核
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 4. 查找轮廓并过滤
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 假设已经通过cv2.findContours获取了contours

        # (1) 打印第2个轮廓的所有点坐标
        print_contour_points(contours, contour_idx=1)

        # (2) 生成可视化图并保存
        vis_img = visualize_contour_points(img, contours, contour_idx=0)

        # 显示图像
        cv2.imshow("Contour Points Visualization", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存图像
        cv2.imwrite("contour_points_visualization.jpg", vis_img)
        polygons = []
        # #取极值点================================================================1,容易形成三角形，勉强
        # contours_4pt = []  # 存储只含四个极值点的新轮廓

        # for i, cnt in enumerate(contours):
        #     try:
        #         pts = np.squeeze(cnt)
        #         if len(pts) < 3:  # 忽略点太少的轮廓
        #             continue
        
        #         # 1. 获取四个极值点（修正语法错误）
        #         extremes = np.array([
        #             pts[np.argmin(pts[:, 0])],  # x_min
        #             pts[np.argmax(pts[:, 1])],  # y_max
        #             pts[np.argmax(pts[:, 0])],  # x_max
        #             pts[np.argmin(pts[:, 1])]   # y_min
        #         ])  # 注意这里只有一个右括号
        
        #             # 2. 闭合多边形（首尾点相同）
        #         closed_extremes = np.vstack([extremes, extremes[0]])
        
        #         # 3. 转换为OpenCV轮廓格式
        #         new_contour = closed_extremes.reshape(-1, 1, 2).astype(np.int32)
        #         contours_4pt.append(new_contour)
        
        #         # 4. 创建Polygon（带坐标转换）
        #         transformed = transform_coords(closed_extremes, img_height=height)
        #         polygons.append(Polygon(transformed))
        
        #         print(f"轮廓{i}: 保留点={len(closed_extremes)}")
        
        #     except Exception as e:
        #         print(f"处理轮廓{i}时出错: {str(e)}")
        #         continue

        # # 替换原始contours前先检查
        # if len(contours_4pt) > 0:
        #     contours = contours_4pt
        #     print(f"成功处理{len(contours)}个轮廓")
        # else:
        #     print("警告：没有生成任何有效轮廓！")
    
        # # 3. 创建Polygon（Shapely会自动处理闭合）
        # transformed = transform_coords(closed_extremes, img_height=height)
        # polygons.append(Polygon(transformed))

        # # 替换原始contours
        # contours = contours_4pt
        # #取极值点================================================================1
        
        contours_4pt = []
        # 在您的轮廓处理循环中：
        # for cnt in contours:
        #     pts = np.squeeze(cnt)
        #     if len(pts) < 3:
        #         continue
            
        #     # 获取视觉特征点（自动处理边缘情况）
        #     visual_points = get_visual_extreme_points(pts, width, height)
            
        #     # 闭合多边形（首尾相连）
        #     closed_points = np.vstack([visual_points, visual_points[0]])
            
        #     # 转换为Polygon
        #     polygon = Polygon(transform_coords(closed_points, height))
        for i, cnt in enumerate(contours):
            pts = np.squeeze(cnt)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 2)  # 确保是 (N, 2)
            if len(pts) < 3:
                continue
            
            
            # 提取极值点（包括边缘点）
            extremes = get_visual_extreme_points(pts, width, height)
            # extremes = get_extreme_points(pts, width, height)

            # 如果不足4个点，补充几何中心
            if len(extremes) < 4:
                center = np.mean(pts, axis=0)
                extremes = np.vstack([extremes, [center]])

            # 闭合多边形
            closed_extremes = np.vstack([extremes, extremes[0]])

            # 转换为OpenCV格式并存储
            contours_4pt.append(closed_extremes.reshape(-1, 1, 2).astype(np.int32))
            polygons.append(Polygon(transform_coords(closed_extremes, height)))
        # 在findContours后立即添加
        print(f"origin Canny contour's number: {len(processed)}")
        print(f"improve Canny contour's number: {len(contours)}")
        print(f"polugons number: {len(polygons)}")
        # 创建纯黑背景
        blank = np.zeros_like(img, dtype=np.uint8)
        blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)  # 转为彩色

        # 绘制轮廓（白色）
        cv2.drawContours(blank, contours_4pt, -1, (255, 255, 255), 1)
        
        cv2.imshow("Contours on Black", blank)
        cv2.waitKey(0)
        # 在findContours后立即添加
        
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            print(f"contour{i}: number of vertex={len(cnt)}, area={area:.1f}")
            
        # polygons = []
        # for cnt in contours:
        #     if len(cnt) >= 4:  # 更严格的顶点数要求
        #         area = cv2.contourArea(cnt)
        #         if area >= min_area:
        #             pts = np.squeeze(cnt)
        #             transformed = transform_coords(pts)
        #             polygon = Polygon(transformed)
        #             if polygon.is_valid:
        #                 simplified = polygon.simplify(simplify_tolerance, preserve_topology=True)
        #                 if not simplified.is_empty:
        #                     polygons.append(simplified)
        for cnt in contours:
                area = cv2.contourArea(cnt)
                pts = np.squeeze(cnt)
                transformed = transform_coords(pts,img_height=height)
                polygon = Polygon(transformed)
                polygons.append(polygon)
                
        # 使用示例
        # print_contour_points(contours)
        
        # 使用示例（检查第一个轮廓）
        # debug_img = visualize_contour_points(img, contours, 0)
        
        # 使用示例
        # visualize_transformed_points(img, contours, height, 0)
        
        # 基本用法：显示所有点和连线
        # vis_img = visualize_contours(img, contours)

        # 高级用法：显示极值点和索引
        vis_img = visualize_contours(img, contours, 
                           highlight_extremes=True, 
                           show_indices=True)
        # 保存结果
        cv2.imwrite(output_contours, vis_img)
        
        if not polygons:
            raise ValueError("invalid polygons")
        
        # 5. 处理坐标系和输出
        output_dir = os.path.dirname(output_shp)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")
        gdf.to_file(f"{output_shp}.shp", encoding='utf-8')
        
        print(f"successfully！生成文件：{output_shp}.shp")
        return True
        
    except Exception as e:
        print(f"transformed failed: {str(e)}")
        return False

# 使用示例（带错误处理）
success = jpg_to_shapefile(
    input_jpg=r"D:\\github\\liumengting\\HBNet\\buildinglab.png",
    output_shp=r"D:\\github\\liumengting\\HBNet\\output12\\building_vector",
    output_contours=r"D:\\github\\liumengting\\HBNet\\selected_contour.jpg",
    threshold=150,
    simplify_tolerance=1.0,
    min_area=50
)

if not success:
    print("请检查输入参数或图像质量")
