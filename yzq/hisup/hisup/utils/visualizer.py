# ux attach -t 0
import os
import cv2
import numpy as np
import os.path as osp
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
from descartes import PolygonPatch
import os
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.transform import from_origin
from rasterio.crs import CRS
from geopandas import GeoSeries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches as Patches
import os
import rasterio
from PIL import Image, ImageDraw
# def save_shapefile_from_polygons(polygons, image_path, output_shapefile_path, resolution=0.1, crs="EPSG:32651"):
#     """
#     将多边形保存为带有地理信息的 Shapefile 文件。

#     参数:
#         polygons (list): 包含 Shapely Polygon 对象的列表。
#         image_path (str): 输入 GeoTIFF 图像的路径，用于获取地理信息。
#         output_shapefile_path (str): 输出 Shapefile 文件的路径。
#         resolution (float): 图像的分辨率（默认为 0.1）。
#         crs (str or CRS): 坐标系（默认为 None，从输入图像中获取）。
#     """
#     # 打开输入图像以获取地理信息
#     with rasterio.open(image_path) as src:
#         left, bottom = src.bounds.left, src.bounds.bottom
#         if crs is None:
#             crs = src.crs  # 从输入图像中获取坐标系
# #     crs = "EPSG:32651"  # WGS 84 / UTM zone 51N
#     # 将多边形转换为地理坐标系
#     polygons_geo = []
#     for poly in polygons:
#         x, y = poly.exterior.coords.xy
#         x_ = [i * resolution + left for i in x]
#         y_ = [j * resolution + bottom for j in y]
#         poly_geo = Polygon(zip(x_, y_))
#         polygons_geo.append(poly_geo)

#     # 创建 GeoDataFrame
#     polygons_gpd = gpd.GeoSeries(polygons_geo)

#     # 翻转多边形（如果需要）
#     origin = ((left + src.bounds.right) / 2, (bottom + src.bounds.top) / 2)
#     flip = GeoSeries.scale(polygons_gpd, xfact=1.0, yfact=-1.0, zfact=0, origin=origin)

#     # 保存为 Shapefile
#     gdf = gpd.GeoDataFrame(crs=crs, geometry=flip)
#     gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')

#     print(f"Shapefile saved to {output_shapefile_path}")
import os
import rasterio
import geopandas as gpd
from shapely.geometry import Polygon
from geopandas import GeoSeries

def save_shapefile_from_polygons(polygons, image_path, output_shapefile_path, resolution=0.1, crs= None):
    """
    将多边形保存为带有地理信息的 Shapefile 文件。

    参数:
        polygons (list): 包含 Shapely Polygon 对象的列表。
        image_path (str): 输入 GeoTIFF 图像的路径，用于获取地理信息。
        output_shapefile_path (str): 输出 Shapefile 文件的路径。
        resolution (float): 图像的分辨率（默认为 0.1）。
        crs (str or CRS): 坐标系（默认为 None，从输入图像中获取）。
    """
    # 检查 polygons 是否为空
    if not polygons:
        print("Warning: No polygons to save. Skipping Shapefile creation.")
        return

    # 检查输入图像路径是否为文件
    if not os.path.isfile(image_path):
        raise ValueError(f"Input image path is not a file: {image_path}")

    # 打开输入图像以获取地理信息
    with rasterio.open(image_path) as src:
        left, bottom = src.bounds.left, src.bounds.bottom
        if crs is None:
            crs = src.crs  # 从输入图像中获取坐标系

    # 将多边形转换为地理坐标系
    polygons_geo = []
    for poly in polygons:
        x, y = poly.exterior.coords.xy
        x_ = [i * resolution + left for i in x]
        y_ = [j * resolution + bottom for j in y]
        poly_geo = Polygon(zip(x_, y_))
        polygons_geo.append(poly_geo)

    # 创建 GeoDataFrame
    polygons_gpd = gpd.GeoSeries(polygons_geo)

    # 翻转多边形（如果需要）
    origin = ((left + src.bounds.right) / 2, (bottom + src.bounds.top) / 2)
    flip = GeoSeries.scale(polygons_gpd, xfact=1.0, yfact=-1.0, zfact=0, origin=origin)

    # 创建 shp 文件夹
    shp_dir = os.path.join(os.getcwd(), "shp")  # 当前目录下的 shp 文件夹
    os.makedirs(shp_dir, exist_ok=True)  # 如果文件夹不存在，则创建

    # 设置输出路径
    output_shapefile_path = os.path.join(shp_dir, os.path.basename(output_shapefile_path))

    # 保存为 Shapefile
    gdf = gpd.GeoDataFrame(crs=crs, geometry=flip)
    gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')

    print(f"Shapefile saved to {output_shapefile_path}")
# set color maps for visulization
# colormap = mpl.cm.Paired.colors
# num_color = len(colormap)

colormap = mpl.cm.Paired.colors
colormap = (
    (0.6509803921568628, 0.807843137254902, 0.8901960784313725), 
    (0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
    (0.984313725490196, 0.6039215686274509, 0.6), 
    (0.8901960784313725, 0.10196078431372549, 0.10980392156862745), 
    (0.9921568627450981, 0.7490196078431373, 0.43529411764705883), 
    (1.0, 0.4980392156862745, 0.0), 
    (0.792156862745098, 0.6980392156862745, 0.8392156862745098), 
    (0.41568627450980394, 0.23921568627450981, 0.6039215686274509), 
    (1.0, 1.0, 0.6), 
    (0.6941176470588235, 0.34901960784313724, 0.1568627450980392))

num_color = len(colormap)


def show_polygons(image, polys):
    plt.axis('off')
    plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.fill(polygon[:,0], polygon[:, 1], color=color, alpha=0.3)
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
    plt.show()

# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)
# #     plt.gca().add_patch(Patches.Polygon(polygon, fill=True, fc=color, ec='black', linewidth=2))

#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon in enumerate(polys):
#         print(f"Polygon {i}: { polygon}")

#         color = colormap[i % num_color]
#         plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec='red', linewidth=2))
#         plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
#     impath = osp.join(save_path, 'viz', 'lyg_test', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)

# #     impath = osp.join(save_path, filename)

# #     plt.savefig(impath, bbox_inches='tight', pad_inches=0.0)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1)

#     plt.clf()
#####################################last############################################
'''
def save_viz(image, polys, save_path, filename):
    plt.axis('off')
    plt.imshow(image)
    print(f"Number of polygons: {len(polys)}")

    for i, polygon_str in enumerate(polys):
        print(f"Polygon {i}: {polygon_str}")

        # 转换多边形字符串为 numpy 数组
        polygon = convert_polygon_str_to_array(polygon_str)
        cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
        for vertex in polygon:
            cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)
#         color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec='red', linewidth=0.01))
#         plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.', markersize=0.05)
        plt.plot(polygon[:,0], polygon[:,1], color='red', marker='.', linestyle='-', linewidth=0.1)
        plt.plot(polygon[:, 0], polygon[:, 1], color='red', marker='.', markersize=2, linestyle=' ', linewidth=0.1)


    impath = osp.join(save_path, 'viz', 'lyg_test4', filename)
    os.makedirs(os.path.dirname(impath), exist_ok=True)
    plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.clf()
'''


# 假设 convert_polygon_str_to_array 和其他参数已经定义
# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)

#         # 确保多边形是闭合的，如果没有闭合，添加第一个点到最后
#         if not np.array_equal(polygon[0], polygon[-1]):
#             polygon = np.vstack([polygon, polygon[0]])

#         # 使用 OpenCV 绘制多边形
#         cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)  # 红色，粗度为2
#         for vertex in polygon:
#             cv2.circle(image, tuple(vertex), 5, (0, 0, 255), -1)  # 顶点为红色，半径为5

#         # 使用 matplotlib 绘制多边形
#         plt.gca().add_patch(Patches.Polygon(polygon, fill=False, edgecolor='red', linewidth=0.5))
#         plt.plot(polygon[:,0], polygon[:,1], color='red', marker='.', linestyle='-', linewidth=0.5)

#     # 保存可视化的图像
#     impath = os.path.join(save_path, 'viz', 'lyg_test4', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()
#################################################################
# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)
        
#         # 检查多边形是否为空
#         if polygon.shape[0] == 0:
#             print(f"Warning: Polygon {i} is empty, skipping.")
#             continue
        
#         cv2.polylines(image, [polygon], isClosed=True, color='green', thickness=1)
#         for vertex in polygon:
#             cv2.circle(image, tuple(vertex),1, 'red', -1)
        
# #         plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec='red', linewidth=1.5))

#     impath = osp.join(save_path, 'viz', 'lyg_test4', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()

# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")
# #         print(f"Polygon shape: {polygon.shape}, dtype: {polygon.dtype}")


#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)
#         print(f"Polygon shape: {polygon.shape}, dtype: {polygon.dtype}")

#         # 绘制多边形
#         cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)
#         for vertex in polygon:
#             cv2.circle(image, tuple(vertex), 1, (0, 0, 255), -1)  # 红色顶点
        
#         # 可选: 使用 matplotlib 绘制边界
# #         plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec='red', linewidth=0.01))
# #         plt.plot(polygon[:, 0], polygon[:, 1], color='red', marker='.', markersize=10, linestyle='-', linewidth=0.1)

#     impath = osp.join(save_path, 'viz', 'lyg_test4', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()
# '''对对对
# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)

#         # 确保 polygon 的类型为 int32 且形状正确
# #         polygon = polygon.astype(np.int32)
# #         polygon = polygon.reshape((-1, 1, 2))  # 转换为 (N, 1, 2) 格式

#         # 检查多边形是否为空
#         if polygon.shape[0] == 0:
#             print(f"Warning: Polygon {i} is empty, skipping.")
#             continue

# #         # 绘制多边形边界
# #         cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)

# #         # 绘制多边形顶点
# #         for vertex in polygon:
# #             cv2.circle(image, tuple(vertex[0]), 1, (255, 0, 0), -1)

# #         # 可选：添加边界框
#         plt.gca().add_patch(Patches.Polygon(polygon.reshape(-1, 2), fill=False, ec='green', linewidth=0.3))
# # #         # 可选: 使用 matplotlib 绘制边界
# #         plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec='red', linewidth=0.01))
# #         plt.plot(polygon[:, 0], polygon[:, 1], color='green', marker='.', markersize=2, linestyle='-', linewidth=0.1)
#         plt.plot(polygon[0, 0], polygon[0, 1], color='red', marker='o', markersize=1)  # 设置更大的点，颜色为红色

#     impath = osp.join(save_path, 'viz', 'lyg_test6', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()
# '''


# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)

#         # 检查多边形是否为空
#         if polygon.shape[0] == 0:
#             print(f"Warning: Polygon {i} is empty, skipping.")
#             continue

#         # 绘制多边形
# #         plt.gca().add_patch(Polygon(polygon, fill=False, edgecolor='green', linewidth=0.3))
#         plt.gca().add_patch(Polygon(polygon, edgecolor='green', linewidth=0.3))

# #         plt.gca().add_patch(Polygon(polygon, facecolor='none', edgecolor='green', linewidth=0.3))

#         # 绘制所有角点
#         for j in range(polygon.shape[0]):
#             plt.plot(polygon[j, 0], polygon[j, 1], color='red', marker='o', markersize=2)  # 使用浮点数坐标

#     impath = os.path.join(save_path, 'viz', 'lyg_test6', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()
def save_viz(image, polys, save_path, filename):
    # 将 NumPy 图像转换为 PIL 图像
    output_image = Image.fromarray(image)

    # 动态计算角点大小
    marker_size = max(1, int(0.0001 * min(image.shape[:2])))

    for polygon_str in polys:
        polygon = convert_polygon_str_to_array(polygon_str)
        if polygon.shape[0] == 0:
            continue

        # 将多边形坐标转换为整数
        polygon = [(int(x), int(y)) for x, y in polygon]

        # 绘制多边形
        draw = ImageDraw.Draw(output_image)
        draw.polygon(polygon, outline="red", width=1)

        # 绘制角点
        for x, y in polygon:
            draw.ellipse([x - marker_size, y - marker_size, x + marker_size, y + marker_size], fill="red")

    # 创建保存路径并保存图像
    impath = os.path.join(save_path, 'viz', 'lyg_test_u8_better', filename)
    os.makedirs(os.path.dirname(impath), exist_ok=True)
    output_image.save(impath, dpi=(1200, 1200))  # 设置高 dpi


def convert_polygon_str_to_array(polygon_str):
    """
    将 polygon 字符串、Polygon 对象或多边形数组转换为 numpy 数组。

    参数:
    - polygon_str: Polygon 对象或多边形坐标的字符串或数组

    返回:
    - numpy 数组，形状为 (N, 2)，每行是一个顶点的 (x, y) 坐标
    """
    # 如果传入的是 Polygon 对象
    if isinstance(polygon_str, Polygon):
        # 获取多边形的外部坐标并转换为 numpy 数组
        coords = np.array(polygon_str.exterior.coords, dtype=np.float32)

    # 如果传入的是字符串，假设它是一个表示多边形的坐标列表
    elif isinstance(polygon_str, str):
        # 尝试将字符串解析为坐标列表
        try:
            # 假设字符串是一个 JSON 格式或 CSV 格式的坐标数据
            coords = np.array(eval(polygon_str), dtype=np.float32)
        except Exception as e:
            print(f"Error parsing polygon string: {e}")
            return np.array([])  # 返回空数组作为错误处理

    # 如果传入的是普通的坐标数组
    elif isinstance(polygon_str, (list, np.ndarray)):
        coords = np.array(polygon_str, dtype=np.float32)
        # 确保数组是二维的，形状为 (N, 2)
        if coords.ndim == 1:
            coords = coords.reshape((-1, 2))

    else:
        print("Error: Invalid input type for polygon_str.")
        return np.array([])  # 返回空数组作为错误处理

    # 确保返回的数组是二维的，形状为 (N, 2)
    if coords.ndim == 1:
        coords = coords.reshape((-1, 2))

    return coords

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as Patches
# import cv2

# def convert_polygon_str_to_array(polygon_str):
#     # 假设 polygon_str 是一个可以转换成 numpy 数组的列表或字符串
#     return np.array(polygon_str, dtype=np.float32).reshape((-1, 2))

# def save_viz(image, polys, save_path, filename, line_color=(0, 255, 0), line_thickness=2, vertex_color=(0, 0, 255), vertex_thickness=5):
#     """
#     在图像上绘制多边形并保存结果。

#     参数：
#     - image (numpy.ndarray): 输入图像。
#     - polys (list): 包含多边形的列表，每个多边形为坐标数组。
#     - save_path (str): 保存可视化结果的文件夹路径。
#     - filename (str): 保存的文件名。
#     - line_color (tuple): 多边形边缘的颜色，默认为绿色。
#     - line_thickness (int): 多边形边缘的线宽，默认为2。
#     - vertex_color (tuple): 顶点的颜色，默认为红色。
#     - vertex_thickness (int): 顶点的厚度，默认为5。
#     """
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)
#         cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
        
#         # 绘制顶点
#         for vertex in polygon:
#             cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)


#     # 确保文件夹存在
#     impath = os.path.join(save_path, 'viz', 'lyg_test3', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)

#     # 保存图像
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()

#     print(f"Saved visualization to {impath}")

# def save_viz(image, polys, save_path, filename, line_color=(0, 255, 0), line_thickness=2, vertex_color=(0, 0, 255), vertex_thickness=5):
#     """
#     在图像上绘制多边形并保存结果。

#     参数：
#     - image (numpy.ndarray): 输入图像。
#     - polys (list): 包含多边形的列表，每个多边形为坐标数组。
#     - save_path (str): 保存可视化结果的文件夹路径。
#     - filename (str): 保存的文件名。
#     - line_color (tuple): 多边形边缘的颜色，默认为绿色。
#     - line_thickness (int): 多边形边缘的线宽，默认为2。
#     - vertex_color (tuple): 顶点的颜色，默认为红色。
#     - vertex_thickness (int): 顶点的厚度，默认为5。
#     """
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)
        
#         # 确保多边形是整数类型
#         polygon = polygon.astype(np.int32)

#         # 绘制多边形
#         cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
        
#         # 绘制顶点
#         for vertex in polygon:
#             cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)

#     # 确保文件夹存在
#     impath = os.path.join(save_path, 'viz', 'lyg_test3', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)
#     if image.dtype != np.uint8:
#         image = np.uint8(image)

#     # 保存图像
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
# #     plt.clf()

#     print(f"Saved visualization to {impath}")
def show_polygons(image, polys):
    plt.axis('off')
    plt.imshow(image)

    for i, polygon in enumerate(polys):
        color = colormap[i % num_color]
        plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
        plt.fill(polygon[:,0], polygon[:, 1], color=color, alpha=0.3)
        plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
    plt.show()

# def save_viz(image, polys, save_path, filename):
#     plt.axis('off')
#     plt.imshow(image)

#     for i, polygon in enumerate(polys):
#         color = colormap[i % num_color]
#         plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=1.5))
#         plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.')
    
#     impath = osp.join(save_path, 'viz', 'lyg_test3', filename)
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.0)
#     plt.clf()
# def save_viz(image, polys, save_path, filename, line_color=(0, 255, 0), line_thickness=2, vertex_color=(0, 0, 255), vertex_thickness=5): 
#     """
#     在图像上绘制多边形并保存结果。

#     参数：
#     - image (numpy.ndarray): 输入图像。
#     - polys (list): 包含多边形的列表，每个多边形为坐标数组。
#     - save_path (str): 保存可视化结果的文件夹路径。
#     - filename (str): 保存的文件名。
#     - line_color (tuple): 多边形边缘的颜色，默认为绿色。
#     - line_thickness (int): 多边形边缘的线宽，默认为2。
#     - vertex_color (tuple): 顶点的颜色，默认为红色。
#     - vertex_thickness (int): 顶点的厚度，默认为5。
#     """
#     plt.axis('off')
#     plt.imshow(image)
#     print(f"Number of polygons: {len(polys)}")

#     for i, polygon_str in enumerate(polys):
#         print(f"Polygon {i}: {polygon_str}")

#         # 转换多边形字符串为 numpy 数组
#         polygon = convert_polygon_str_to_array(polygon_str)
        
#         # 确保多边形是整数类型并裁剪坐标
#         polygon = np.clip(polygon, 0, [image.shape[1]-1, image.shape[0]-1])
#         polygon = polygon.astype(np.int32)

#         # 绘制多边形
#         cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
        
#         # 绘制顶点
#         for vertex in polygon:
#             cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)

#     # 确保文件夹存在
#     impath = os.path.join(save_path, 'viz', 'lyg_test3', filename)
#     os.makedirs(os.path.dirname(impath), exist_ok=True)

#     # 保存图像
#     plt.savefig(impath, bbox_inches='tight', pad_inches=0.1, dpi=300)
#     plt.clf()

#     print(f"Saved visualization to {impath}")

# # 辅助函数，用于将多边形字符串转换为 numpy 数组
# def convert_polygon_str_to_array(polygon_str):
#     if isinstance(polygon_str, Polygon):
#     # 如果是 Polygon 对象，转换为 WKT 字符串
#         polygon_str = polygon_str.wkt
# #     polygon_str = polygon_str.replace("POLYGON (", "").replace(")", "").replace(" ", "")
# #     coords = polygon_str.split(",")
# #     return np.array([[float(coord.split()[0]), float(coord.split()[1])] for coord in coords])
#     coords = polygon_str.strip('()').split(',')  # 移除括号并分割坐标
#     coords = [coord.strip() for coord in coords]  # 去掉每个坐标的空格
#     for coord in coords:
#         try:
#             x, y = coord.split()
#             float_x = float(x)
#             float_y = float(y)
#         except ValueError:
#             print(f"Invalid coordinate: {coord}")
#     return np.array([[float(coord.split()[0]), float(coord.split()[1])] for coord in coords])
# def convert_polygon_str_to_array(polygon_str):
#     if isinstance(polygon_str, Polygon):
#         # 如果是 Polygon 对象，转换为 WKT 字符串
#         polygon_str = polygon_str.wkt

#     # 去除括号并按逗号分割坐标
#     coords = polygon_str.strip('()').split(',')
#     coords = [coord.strip() for coord in coords]  # 去掉每个坐标的空格

#     # 用于存储最终的坐标对
#     polygon_array = []

#     # 遍历所有坐标
#     for coord in coords:
#         try:
#             # 拆分坐标
#             x, y = coord.split()
#             # 转换为浮动数
#             float_x = float(x)
#             float_y = float(y)
#             polygon_array.append([float_x, float_y])
#         except ValueError:
#             # 如果出现转换错误，打印并跳过
#             print(f"Invalid coordinate: {coord}")

#     # 返回转换后的 numpy 数组
#     return np.array(polygon_array)

# def convert_polygon_str_to_array(polygon_str):
#     """
#     将polygon字符串或Polygon对象转换为numpy数组。
#     """
#     # 如果传入的是 Polygon 对象
#     if isinstance(polygon_str, Polygon):
#         # 获取多边形的外部坐标并转换为numpy数组
#         coords = np.array(polygon_str.exterior.coords, dtype=np.float32)
#     else:
#         # 如果传入的是字符串或其他格式的多边形，直接转换
#         coords = np.array(polygon_str, dtype=np.float32).reshape((-1, 2))

#     return coords

def viz_inria(image, polygons, output_dir, file_name, alpha=0.5, linewidth=12, markersize=45):
    plt.rcParams['figure.figsize'] = (500,500)
    plt.rcParams['figure.dpi'] = 10
    plt.axis('off')
    plt.imshow(image)
    for n, poly in enumerate(polygons):
        poly_color = colormap[n%num_color]
        if poly.type == 'MultiPolygon':
            for p in poly:
                patch = PolygonPatch(p.buffer(0), ec=poly_color, fc=poly_color, alpha=alpha, linewidth=linewidth)
                plt.gca().add_patch(patch)
                plt.gca().add_patch(Patches.Polygon(p.exterior.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                juncs = np.array(p.exterior.coords[:-1])
                plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
                if len(p.interiors) != 0:
                    for inter in p.interiors:
                        plt.gca().add_patch(Patches.Polygon(inter.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                        juncs = np.array(inter.coords[:-1])
                        plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
        else:
            try:
                patch = PolygonPatch(poly.buffer(0), ec=poly_color, fc=poly_color, alpha=alpha, linewidth=linewidth)
                plt.gca().add_patch(patch)
            except TypeError:
                plt.gca().add_patch(Patches.Polygon(poly.exterior.coords[:-1], fill=True, ec=poly_color, fc=poly_color, linewidth=linewidth, alpha=alpha))
            plt.gca().add_patch(Patches.Polygon(poly.exterior.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
            juncs = np.array(poly.exterior.coords[:-1])
            plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
            if len(poly.interiors) != 0:
                for inter in poly.interiors:
                    plt.gca().add_patch(Patches.Polygon(inter.coords[:-1], fill=False, ec=poly_color, linewidth=linewidth))
                    juncs = np.array(inter.coords[:-1])
                    plt.plot(juncs[:,0], juncs[:,1], color=poly_color, marker='.', markersize=markersize, linestyle='none')
    
    # save_filename = os.path.join(output_dir, 'inria_viz', file_name[:-4] + '.svg')
    # plt.savefig(save_filename, bbox_inches='tight', pad_inches=0.0)
    # plt.clf()
    plt.show()


def draw_predictions_with_mask_inria(img, junctions, polys_ids, save_dir, filename):
    plt.axis('off')
    plt.imshow(img)

    instance_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    for i, contours in enumerate(polys_ids):
        color = colormap[i % num_color]
        for h, idx in enumerate(contours):
            poly = junctions[idx]
            if h == 0:
                cv2.drawContours(instance_mask, [np.int0(poly).reshape(-1, 1, 2)], -1, color=color, thickness=-1)
            else:
                cv2.drawContours(instance_mask, [np.int0(poly).reshape(-1, 1, 2)], -1, color=0, thickness=-1)

            plt.gca().add_patch(Patches.Polygon(poly, fill=False, ec=color, linewidth=2))
    
    alpha_map = np.bitwise_or(instance_mask[:,:,0:1].astype(bool), 
                              instance_mask[:,:,1:2].astype(bool), 
                              instance_mask[:,:,2:3].astype(bool)).astype(np.float32)
    instance_mask = np.concatenate((instance_mask, alpha_map), axis=-1)
    plt.imshow(instance_mask, alpha=0.3)
    plt.show()


def draw_predictions_inria(img, junctions, polys_ids):
    plt.axis('off')

    plt.imshow(img)
    for i, contours in enumerate(polys_ids):
        color = colormap[i % num_color]
        for idx in contours:
            poly = junctions[idx]
            plt.gca().add_patch(Patches.Polygon(poly, fill=False, ec=color, linewidth=1.5))
            plt.plot(poly[:,0], poly[:,1], color=color, marker='.')
    plt.show()




