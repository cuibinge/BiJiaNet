import numpy as np
import cv2
import tifffile as tiff
import svgwrite
from PIL import Image
from io import BytesIO
import base64

# -------------------------- 配置参数 --------------------------
CLASS_COLORS = {
    0: "#FF0000",  # 红色
    1: "#00FF00",  # 绿色
    2: "#0000FF",  # 蓝色
    3: "#FFFF00",  # 黄色
    4: "#FF00FF"  # 品红
}

MORPH_KERNEL = np.ones((5, 5), np.uint8)  # 形态学核
OPEN_ITER = 2  # 开运算迭代次数
CLOSE_ITER = 4  # 闭运算迭代次数
STROKE_WIDTH = 1.5


# -------------------------- 核心处理逻辑 --------------------------
def main():
    # 加载分割结果和原始图像
    segmentation = tiff.imread('C:/Users/zzc/Desktop/other/result/5/225.tif').astype(np.uint8)
    target_image = process_image(
        tiff.imread('C:/Users/zzc/Desktop/other/datasets/jiaozhouwan/512/img/225.tif'))

    # 创建SVG画布
    dwg = svgwrite.Drawing(
        '225_img.svg',
        size=(target_image.shape[1], target_image.shape[0]),
        profile='tiny'
    )

    # 添加背景图
    add_background(dwg, target_image)

    # 处理每个类别（跳过类别0）
    for class_id, color in CLASS_COLORS.items():
        if class_id == 0:
            continue  # 跳过类别0
        process_class(dwg, segmentation, class_id, color)

    dwg.save()
    print("多类别SVG生成成功（已去掉类别0）！")


def process_image(img):
    """图像预处理"""
    if img.dtype == np.uint16:
        img = cv2.convertScaleAbs(img, alpha=255.0 / 65535)
    elif img.dtype in [np.float32, np.float64]:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def add_background(dwg, image):
    """添加背景图像"""
    buffered = BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    dwg.add(dwg.image(
        href=f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}",
        insert=(0, 0),
        size=(image.shape[1], image.shape[0])
    ))


def process_class(dwg, segmentation, class_id, color):
    """处理单个类别"""
    # 生成类别掩膜
    class_mask = np.where(segmentation == class_id, 255, 0).astype(np.uint8)

    # 形态学优化
    opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=OPEN_ITER)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=CLOSE_ITER)

    # 提取轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 绘制所有轮廓
    for contour in contours:
        points = contour.squeeze()
        if len(points) < 3:  # 过滤无效轮廓
            continue

        # 生成路径数据
        path_data = "M " + " L ".join([f"{x},{y}" for (x, y) in points]) + " Z"
        dwg.add(dwg.path(
            d=path_data,
            fill="none",
            stroke=color,
            stroke_width=STROKE_WIDTH,
            stroke_linejoin="round"
        ))


if __name__ == "__main__":
    main()
