import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import tifffile as tiff

# 读取真值图
true_value_map = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\result\\CMTFNet\\129.tif')

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
    # 根据轮廓周长动态调整简化程度（系数越大，点越少）
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_contours.append(approx)

# 转换坐标格式并收集边缘点
edge_points = []
for cnt in approx_contours:
    cnt_array = cnt.squeeze()
    for point in cnt_array:
        x, y = point  # OpenCV返回的坐标为(x, y)
        edge_points.append([y, x])  # 转换为(y, x)格式

edge_points = np.array(edge_points)

# 保存坐标
with open('edge_points.json', 'w') as f:
    json.dump(edge_points.tolist(), f)

# 读取目标图像并绘制边缘点
target_image = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\128\\img\\129.tif')
for y, x in edge_points:
    cv2.circle(target_image, (x, y), 1, (0, 0, 255), -1)

# 保存结果
cv2.imwrite('result_image129.png', target_image)
plt.imshow(target_image)
plt.axis('off')
plt.show()