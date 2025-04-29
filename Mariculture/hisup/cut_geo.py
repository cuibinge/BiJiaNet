import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def save_stretched_image(image, output_path, meta):
    """
    保存拉伸后的图像，并保留地理坐标信息。

    参数:
        image (numpy.ndarray): 拉伸后的图像（uint8 类型）。
        output_path (str): 输出图像路径。
        meta (dict): 原始图像的元数据。
    """
    # 更新元数据以匹配拉伸后的图像
    meta.update({
        'dtype': 'uint8',  # 更新数据类型为 uint8
        'count': image.shape[2]  # 更新波段数量
    })

    # 保存拉伸后的图像
    with rasterio.open(output_path, 'w', **meta) as dst:
        for channel in range(image.shape[2]):
            dst.write(image[:, :, channel], channel + 1)

def truncate_stretch(image, min_percentile=2, max_percentile=98):
    stretched_channels = []
    image_n0 = image[np.any(image != 0, axis=-1)]
    for channel in range(image.shape[2]):
        image_channel = image[:, :, channel]
        min_val = np.percentile(image_n0[:, channel], min_percentile)
        max_val = np.percentile(image_n0[:, channel], max_percentile)
        print(f"Channel {channel}: min={min_val}, max={max_val}")

        clipped_channel = np.clip(image_channel, min_val, max_val)
        stretched_channel = ((clipped_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        stretched_channels.append(stretched_channel)

    stretched_image = np.stack(stretched_channels, axis=2)
    return stretched_image

# 在 cut_image_with_geoinfo 函数中增加调试信息
def cut_image_with_geoinfo(image_path, label_path, block_size=2048, stride=1024, save_img_path=None, save_label_path=None, sample_threshold=0.1):
    with rasterio.open(image_path) as src_img, rasterio.open(label_path) as src_label:
        img_meta = src_img.meta
        label_meta = src_label.meta

        image = src_img.read()
        image = np.transpose(image, (1, 2, 0))
        image = image[:,:,[2,1,0]]

        print(f"Image shape: {image.shape}")

        stretched_image = truncate_stretch(image)
#         stretched_image_path = "./data/lyg_3/geo/stretched_image.tif"
#         save_stretched_image(stretched_image, stretched_image_path, img_meta)
#         print(f"Stretched image saved to {stretched_image_path}")

        height, width = image.shape[:2]
        num = 0

        for i in tqdm(range(0, height, stride), desc="Processing rows"):
            for j in range(0, width, stride):
                row_start = i
                col_start = j
                row_end = min(i + block_size, height)
                col_end = min(j + block_size, width)

                if row_end - row_start < block_size or col_end - col_start < block_size:
                    row_start = max(0, row_end - block_size)
                    col_start = max(0, col_end - block_size)

                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                print(f"Window: row_start={row_start}, col_start={col_start}, row_end={row_end}, col_end={col_end}")

#                 img_block = src_img.read(window=window)
                img_block = stretched_image[row_start:row_end, col_start:col_end, :]

                label_block = src_label.read(window=window)

                sample_ratio = np.sum(label_block > 0) / label_block.size
                if sample_ratio < sample_threshold:
                    continue

                img_meta.update({
                    'height': block_size,
                    'width': block_size,
                    'transform': rasterio.windows.transform(window, src_img.transform)
                })
                label_meta.update({
                    'height': block_size,
                    'width': block_size,
                    'transform': rasterio.windows.transform(window, src_label.transform)
                })

                if save_img_path:
                    output_img_path = os.path.join(save_img_path, f"img_{num}.tif")
                    with rasterio.open(output_img_path, 'w', **img_meta) as dst_img:
                        # 转置数据
                        data_to_write = np.transpose(img_block, (2, 0, 1))

                        # 检查波段数量和数据形状
                        print("Source shape:", data_to_write.shape)
                        print("Destination bands:", dst_img.count)

                        # 写入数据
                        dst_img.write(data_to_write, indexes=[1, 2, 3])


                if save_label_path:
                    output_label_path = os.path.join(save_label_path, f"img_{num}.tif")
                    with rasterio.open(output_label_path, 'w', **label_meta) as dst_label:
                        dst_label.write(label_block)

                num += 1

# 输入图像和标签路径
image_path = "./data/dongtou/img/GF2_PMS2_E121.0_N27.9_20230216_L1A0007111740-MSS2+PAN1.tif"
label_path = "./data/dongtou/gt/GF2_PMS2_E121.0_N27.9_20230216_L1A0007111740-MSS2+PAN1.tif"

# 保存拉伸后图像的路径
stretched_image_dir = "./data/dongtou/geo/stretched/"
os.makedirs(stretched_image_dir, exist_ok=True)

# 保存裁切后图像和标签的路径
save_img_path = "./data/dongtou/geo/cut_2048_u8_new/img/"
save_label_path = "./data/dongtou/geo/cut_2048_u8_new/gt/"

# 创建保存目录
os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_label_path, exist_ok=True)

# 调用裁切函数
cut_image_with_geoinfo(
    image_path, label_path,
    block_size=2048, stride=1024,
    save_img_path=save_img_path, save_label_path=save_label_path,
    sample_threshold=0.1  # 样本比例阈值设置为 20%
)
