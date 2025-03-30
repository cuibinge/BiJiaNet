import numpy as np
import cv2
import tifffile as tiff
import svgwrite
from PIL import Image
from io import BytesIO
import base64

# 读取真值图
true_value_map = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\result\\CMTFNet\\img1.tif')

# Sobel梯度计算
gradient_x = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(true_value_map.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
gradient = np.abs(gradient_x) + np.abs(gradient_y)

# 阈值提取边缘
threshold = 0.5
edge_map = gradient > threshold
edge_map_uint8 = np.uint8(edge_map * 255)

# 查找并简化轮廓
contours, _ = cv2.findContours(edge_map_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
approx_contours = []
for contour in contours:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_contours.append(approx)

# 读取目标图像并处理
target_image = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\yuantu\\img\\img.tif')

# --- 关键修正：确保图像数据正确 ---
# 1. 处理数据类型和范围
if target_image.dtype == np.float32 or target_image.dtype == np.float64:
    # 假设浮点型数据范围为 [0,1]
    target_image = (target_image * 255).astype(np.uint8)
elif target_image.dtype == np.uint16:
    # uint16 转 uint8（按比例缩放）
    target_image = cv2.convertScaleAbs(target_image, alpha=(255.0/65535.0))
else:
    target_image = target_image.astype(np.uint8)

# 2. 确保通道顺序为RGB
if len(target_image.shape) == 2:
    target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)
elif target_image.shape[2] == 3:
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR转RGB

# 将图像转换为Base64
buffered = BytesIO()
Image.fromarray(target_image).save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

# 创建SVG画布
dwg = svgwrite.Drawing(
    'vector_overlay_129.svg',
    size=(target_image.shape[1], target_image.shape[0]),
    profile='tiny'
)

# 添加Base64背景图像
dwg.add(dwg.image(
    href=f"data:image/png;base64,{img_base64}",
    insert=(0, 0),
    size=(target_image.shape[1], target_image.shape[0])
))

# 绘制矢量线段（按从左到右、从上到下排序）
y_tolerance = 5
stroke_color = "green"
stroke_width = 1

for cnt in approx_contours:
    if len(cnt) < 2:
        continue
    points = cnt.squeeze().tolist()
    points_sorted = sorted(points, key=lambda p: (p[1] // y_tolerance, p[0]))
    path_data = "M " + " L ".join([f"{x},{y}" for (x, y) in points_sorted])
    dwg.add(dwg.path(d=path_data, fill="none", stroke=stroke_color, stroke_width=stroke_width))

# 保存SVG
dwg.save()
print("SVG生成成功！")