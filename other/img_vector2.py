import numpy as np
import cv2
import tifffile as tiff
import svgwrite
from PIL import Image
from io import BytesIO
import base64

# -------------------------- 修改1: 读取分割结果而非真值图 --------------------------
# 假设你的分割结果是二值图像，文件路径可能需要调整
segmentation_result = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\result\\CMTFNet\\img1.tif').astype(np.uint8)

# -------------------------- 修改2: 添加形态学处理 --------------------------
# 1. 开运算去噪
kernel = np.ones((3,3), np.uint8)
opened = cv2.morphologyEx(segmentation_result, cv2.MORPH_OPEN, kernel, iterations=1)

# 2. 闭运算连接断裂
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

# -------------------------- 修改3: 使用RETR_EXTERNAL提取最外层轮廓 --------------------------
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# -------------------------- 修改4: 筛选最大轮廓 --------------------------
if not contours:
    raise ValueError("未检测到有效轮廓")

max_contour = max(contours, key=cv2.contourArea)

# -------------------------- 修改5: 优化轮廓平滑参数 --------------------------
epsilon = 0.005 * cv2.arcLength(max_contour, True)  # 减小epsilon系数使边界更平滑
approx = cv2.approxPolyDP(max_contour, epsilon, True)

# 读取目标图像并处理（保持你的原始逻辑）
target_image = tiff.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\yuantu\\img\\img.tif')

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

# -------------------------- 修改6: 优化SVG绘制 --------------------------
dwg = svgwrite.Drawing(
    'vector_overlay_img.svg',
    size=(target_image.shape[1], target_image.shape[0]),
    profile='tiny'
)

# 添加背景图
dwg.add(dwg.image(
    href=f"data:image/png;base64,{img_base64}",
    insert=(0, 0),
    size=(target_image.shape[1], target_image.shape[0])
))

# -------------------------- 修改7: 直接绘制平滑后的最大轮廓 --------------------------
stroke_color = "green"
stroke_width = 1.5  # 适当加粗线条

# 将轮廓坐标转换为SVG路径（保留原始顺序）
path_data = "M " + " L ".join([f"{x},{y}" for (x, y) in approx.squeeze()])
path = dwg.path(d=path_data, fill="none", stroke=stroke_color, stroke_width=stroke_width)
path.push("Z")  # 闭合路径
dwg.add(path)

dwg.save()
print("SVG生成成功！")