import numpy as np
import cv2
import tifffile as tiff
import svgwrite
from PIL import Image
from io import BytesIO
import base64





# -------------------------- 修改1: 调整形态学核大小 --------------------------
segmentation_result = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\result\\CMTF_SRAM\\226.tif').astype(np.uint8)

# 使用更大的核处理超大图像
kernel = np.ones((5,5), np.uint8)  # 原为(3,3)
opened = cv2.morphologyEx(segmentation_result, cv2.MORPH_OPEN, kernel, iterations=2)  # 增加迭代次数
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=4)  # 加强连接

# -------------------------- 修改2: 禁用多边形近似 --------------------------
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 使用NONE模式保留所有点

if not contours:
    raise ValueError("未检测到有效轮廓")

max_contour = max(contours, key=cv2.contourArea)

# -------------------------- 修改3: 直接使用原始轮廓点（不简化） --------------------------
raw_points = max_contour.squeeze()

# -------------------------- 处理目标图像（保持原逻辑） --------------------------
target_image = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\512\\img\\226.tif')

# --- 数据类型和通道处理 ---
if target_image.dtype == np.float32 or target_image.dtype == np.float64:
    target_image = (target_image * 255).astype(np.uint8)
elif target_image.dtype == np.uint16:
    target_image = cv2.convertScaleAbs(target_image, alpha=(255.0/65535.0))
else:
    target_image = target_image.astype(np.uint8)

if len(target_image.shape) == 2:
    target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)
elif target_image.shape[2] == 3:
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

# 转换为Base64
buffered = BytesIO()
Image.fromarray(target_image).save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

# -------------------------- 修改4: 优化SVG路径生成 --------------------------
dwg = svgwrite.Drawing(
    'vector_overlay_226_3.svg',
    size=(target_image.shape[1], target_image.shape[0]),
    profile='tiny'
)

# 添加背景图（保持原逻辑）
dwg.add(dwg.image(
    href=f"data:image/png;base64,{img_base64}",
    insert=(0, 0),
    size=(target_image.shape[1], target_image.shape[0])
))

# -------------------------- 修改5: 分段绘制路径 --------------------------
stroke_color = "green"
stroke_width = 1.5
chunk_size = 1000  # 分段处理避免内存问题

# 方法1：直接绘制所有点（适合高性能设备）
path_data = "M " + " L ".join([f"{x},{y}" for (x, y) in raw_points])
path = dwg.path(d=path_data, fill="none", stroke=stroke_color, stroke_width=stroke_width)
path.push("Z")
dwg.add(path)

# 方法2：分段绘制（内存优化版）
"""
for i in range(0, len(raw_points), chunk_size):
    chunk = raw_points[i:i+chunk_size]
    path_data = "M " + " L ".join([f"{x},{y}" for (x, y) in chunk])
    dwg.add(dwg.path(d=path_data, fill="none", stroke=stroke_color, stroke_width=stroke_width))
"""

dwg.save()
print("SVG生成成功！")