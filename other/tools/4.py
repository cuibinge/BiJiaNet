import numpy as np
import cv2
import tifffile as tiff
import svgwrite
from PIL import Image

# 读取真值图
true_value_map = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\result\\CMTFNet\\129.tif')

# Sobel梯度计算
gradient_x = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
gradient = np.abs(gradient_x) + np.abs(gradient_y)

# 阈值提取边缘
threshold = 0.5
edge_map = gradient > threshold
edge_map_uint8 = np.uint8(edge_map * 255)

# 查找并简化轮廓
contours, _ = cv2.findContours(edge_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
approx_contours = []
for contour in contours:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_contours.append(approx)

# 读取目标图像并转RGB
target_image = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\128\\img\\129.tif')
if len(target_image.shape) == 2:
    target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)

# 保存背景为PNG
background_path = "target_background.png"
Image.fromarray(target_image).save(background_path)

# 创建SVG画布
dwg = svgwrite.Drawing('vector_overlay_129_new.svg', size=(target_image.shape[1], target_image.shape[0]))
dwg.add(dwg.image(href=background_path, insert=(0, 0), size=(target_image.shape[1], target_image.shape[0])))

# 定义线段参数
stroke_color = "green"
stroke_width = 1
y_tolerance = 5  # 行分组容差

# 按从左到右、从上到下排序绘制
for cnt in approx_contours:
    if len(cnt) < 2:
        continue
    points = cnt.squeeze().tolist()

    # 关键排序：先按行分组，再按X坐标排序
    points_sorted = sorted(points, key=lambda p: (p[1] // y_tolerance, p[0]))

    # 生成路径
    path_data = "M " + " L ".join([f"{x},{y}" for (x, y) in points_sorted])
    dwg.add(dwg.path(d=path_data, fill="none", stroke=stroke_color, stroke_width=stroke_width))

# 保存SVG
dwg.save()