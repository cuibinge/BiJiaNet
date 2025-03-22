import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import tifffile as tiff

# 读取真值图
true_value_map = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\result\\CMTFNet\\72.tif')

# 使用 Sobel 算子计算梯度（保持原始尺寸）
gradient_x = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
gradient = np.abs(gradient_x) + np.abs(gradient_y)

# 设定阈值提取边缘
threshold = 0.5
edge_map = gradient > threshold

# 将边缘图转换为二值图像
edge_map_uint8 = np.uint8(edge_map * 255)

# 查找轮廓并简化
contours, _ = cv2.findContours(edge_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
approx_contours = []

for contour in contours:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_contours.append(approx)

# 读取目标图像
target_image = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\128\\img\\72.tif')

# 绘制连线和点
for cnt in approx_contours:
    # 转换为图像坐标系 (x, y)
    cnt_points = cnt.squeeze()

    # 绘制线段（连轮廓点）
    cv2.polylines(target_image, [cnt], isClosed=False, color=(0, 255, 0), thickness=1)  # 绿色连线

    # 可选：绘制点（保留原红色点）
    for point in cnt_points:
        x, y = point
        cv2.circle(target_image, (x, y), 1, (0, 0, 255), -1)  # 红色点

# 保存坐标（如果需要）
edge_points = [cnt.squeeze().tolist() for cnt in approx_contours]
with open('edge_points.json', 'w') as f:
    json.dump(edge_points, f)

# 保存结果
cv2.imwrite('result_image.png', target_image)
plt.imshow(target_image)
plt.axis('off')
plt.show()