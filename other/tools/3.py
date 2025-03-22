import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
import geojson


def mask_to_geojson(mask, transform, output_file):
    """
    将二分类分割掩膜转换为GeoJSON格式的矢量图
    :param mask: 二维数组，语义分割结果掩膜（0和1的值）
    :param transform: rasterio的仿射变换矩阵，用于将像素坐标转换为地理坐标
    :param output_file: 输出的GeoJSON文件路径
    """
    # 提取掩膜中的边界
    shapes = features.shapes(mask, transform=transform)

    # 过滤出类别为1的区域
    filtered_shapes = [geom for geom, value in shapes if value == 1]

    # 创建GeoJSON FeatureCollection
    features = []
    for geom in filtered_shapes:
        feature = geojson.Feature(geometry=geom, properties={"class": 1})
        features.append(feature)

    feature_collection = geojson.FeatureCollection(features)

    # 保存为GeoJSON文件
    with open(output_file, 'w') as f:
        geojson.dump(feature_collection, f)


# 示例用法
if __name__ == "__main__":
    # 假设你的分割结果图是一个二维数组，保存为TIF文件
    mask_path = "segmentation_result.tif"
    output_geojson = "segmentation_result.geojson"

    # 读取分割结果图
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # 读取第一个波段
        transform = src.transform  # 获取仿射变换矩阵

    # 转换为GeoJSON
    mask_to_geojson(mask, transform, output_geojson)
    print(f"矢量图已保存到 {output_geojson}")