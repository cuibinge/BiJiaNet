import numpy as np
import cv2
import tifffile as tiff
import svgwrite
from PIL import Image  # 用于TIFF转PNG

# 读取真值图
true_value_map = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\result\\CMTFNet\\129.tif')

# Sobel梯度计算（保持原始尺寸）
gradient_x = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
gradient = np.abs(gradient_x) + np.abs(gradient_y)

# 设定阈值提取边缘
threshold = 0.5
edge_map = gradient > threshold

# 转换为二值图像
edge_map_uint8 = np.uint8(edge_map * 255)

# 查找轮廓并简化
contours, _ = cv2.findContours(edge_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
approx_contours = []
for contour in contours:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_contours.append(approx)

# 读取目标图像并转换为RGB（假设是单通道，需要转为3通道）
target_image = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\128\\img\\129.tif')
if len(target_image.shape) == 2:
    target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)

# 保存目标图像为PNG（SVG不支持直接嵌入TIFF）
background_path = "target_background.png"
Image.fromarray(target_image).save(background_path)

# 创建SVG画布（尺寸与图像一致）
dwg = svgwrite.Drawing('vector_overlay_129.svg', size=(target_image.shape[1], target_image.shape[0]))

# 添加背景图像到SVG
dwg.add(dwg.image(href=background_path, insert=(0, 0), size=(target_image.shape[1], target_image.shape[0])))

# 定义矢量线段参数
stroke_color = "green"  # 线段颜色
stroke_width = 1     # 线宽

# 遍历轮廓并添加矢量路径
for cnt in approx_contours:
    if len(cnt) < 2:
        continue  # 跳过单点轮廓
    # 转换坐标为SVG路径格式
    path_data = "M " + " L ".join([f"{point[0][0]},{point[0][1]}" for point in cnt])
    # 添加路径到SVG
    dwg.add(dwg.path(d=path_data, fill="none", stroke=stroke_color, stroke_width=stroke_width))

# 保存SVG文件
dwg.save()