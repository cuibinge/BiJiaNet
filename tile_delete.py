import os
import shutil
import numpy as np
import rasterio

# 设置源文件夹
tile_dir = 'inputs/4096/new_image'  # 瓦片图目录
mask_dir = 'inputs/4096/new_label'  # 掩码图目录

# 设置保存保留文件的新文件夹
tile_save_dir = 'inputs/4096/new_image_delete'
mask_save_dir = 'inputs/4096/new_label_delete'

# 创建保存目录（如果不存在的话）
os.makedirs(tile_save_dir, exist_ok=True)
os.makedirs(mask_save_dir, exist_ok=True)

# 列出所有掩码图
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]

for mask_file in mask_files:
    mask_path = os.path.join(mask_dir, mask_file)

    # 读取掩码图
    with rasterio.open(mask_path) as src:
        mask_data = src.read()

    # 判断掩码是否全黑（全0）
    if np.all(mask_data == 0):
        print(f'掩码全黑，跳过：{mask_file}')
    else:
        print(f'掩码正常，保留：{mask_file}')

        # 搬运掩码图
        shutil.copy(mask_path, os.path.join(mask_save_dir, mask_file))

        # 搬运对应的瓦片图
        tile_path = os.path.join(tile_dir, mask_file)
        if os.path.exists(tile_path):
            shutil.copy(tile_path, os.path.join(tile_save_dir, mask_file))
            print(f'瓦片也保留：{mask_file}')
        else:
            print(f'对应瓦片不存在：{mask_file}')
