"""
# @time:2023/3/26 16:28
# Author:Tuan
# @File:cut.py
"""

import glob
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 根目录为当前文件所在目录
root = os.getcwd()


# ====================================================================================================================================


def cut_image(image, image_name, block_size, stride, save_path):
    row = image.shape[0]
    col = image.shape[1]
    dep = image.shape[2]
    # if row % block_size != 0 or col % block_size != 0:
    print("Need padding the image...")
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) * block_size
    padding_w = (col // block_size + 1) * block_size
    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype="float32")
    padding_img[:row, :col, :] = image[:row, :col, :]

    row_num = 0
    for i in tqdm.tqdm(list(np.arange(0, row, stride))):
        row_num += 1

        if (i + block_size) > row:
            continue

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1

            if (j + block_size) > col:
                continue

            block = np.array(padding_img[i: i + block_size, j: j + block_size, :])
            block_name = (
                    image_name + "_" + str(int(row_num)) + "_" + str(int(col_num)) + ".tif"
            )

            # print(block.shape)
            # print(block.min())
            # print(block.max())
            block = block.astype(np.float16)
            imageio.imwrite(os.path.join(save_path, block_name), block)


def cut_label(image, image_name, block_size, stride, save_path):
    row = image.shape[0]
    col = image.shape[1]

    # if row % block_size != 0 or col % block_size != 0:
    print("Need padding the image...")
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) * block_size
    padding_w = (col // block_size + 1) * block_size
    # 初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    padding_pre = np.zeros((padding_h, padding_w), dtype="uint8")
    padding_pre[:row, :col] = image[:row, :col]

    row_num = 0
    for i in list(np.arange(0, row, stride)):
        row_num += 1

        if (i + block_size) > row:
            continue

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1

            if (j + block_size) > col:
                continue

            block = np.array(padding_pre[i: i + block_size, j: j + block_size])
            #            if np.min(block) == 0:
            #                print('ok')
            block_name = (
                    image_name + "_" + str(int(row_num)) + "_" + str(int(col_num)) + ".tif"
            )
            block[block == 1] = 255
            # print(block.shape)
            # print(block.min())
            # print(block.max())
            # print(block)
            # block = block.astype(np.uint8) * 255
            # print(block)

            imageio.imwrite(os.path.join(save_path, block_name), block)


# ====================================================================================================================================

"""
生成彩色图像，为了标记样本
"""

# image 文件路径
# image_path = os.path.join(root, 'lable','xisha','12','train','train_image')
# image_path = r"./data_my/raw_6/img"

image_path = r"D:\desktop\马尾藻数据\tif原图"
image_path_list = glob.glob(os.path.join(image_path, "*.tif"))

# label 文件路径
# label_path = os.path.join(root, "lable", "xisha", "12", "train", "train_label")
label_path = r"D:\desktop\马尾藻数据\tif掩码\label_tif"
label_path_list = glob.glob(os.path.join(label_path, "*.tif"))


# for i in range(len(image_path_list)):
for i in range(len(label_path_list)):

    # 整幅图像的数据和标签
    image_data = imageio.imread(image_path_list[i])  # /1.tif
    label_data = imageio.imread(label_path_list[i])

    #     print(image_data)
    #     print(image_data.dtype)
    #     image_data=image_data.astype(np.uint16)
    #     print("--------------------------------")
    #     print(image_data.dtype)
    #     print(image_data)
    #     B1, B2, B3, B4 = cv2.split(image_data)
    # B1=image_data[:,:,0]
    # B2=image_data[:,:,1]
    # B3=image_data[:,:,2]
    # # B4=image_data[:,:,3]
    # B1_normalization = ((B1 - np.min(B1)) / (np.max(B1) - np.min(B1)) * 1).astype('float32')
    # B2_normalization = ((B2 - np.min(B2)) / (np.max(B2) - np.min(B2)) * 1).astype('float32')
    # B3_normalization = ((B3 - np.min(B3)) / (np.max(B3) - np.min(B3)) * 1).astype('float32')
    # # B4_normalization = ((B4 - np.min(B4)) / (np.max(B4) - np.min(B4)) * 1).astype('float32')
    # # image_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization, B4_normalization])
    # image_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization])
    # 创建切割后的'image'保存文件夹
    # train_img_path = os.path.join(root, 'lable', 'small', 'train','cut_train_images')
    # print(image_path_list[i])

    train_img_path = r"inputs/4096/new_image"
    if not os.path.exists(train_img_path):
        os.makedirs(train_img_path)

    # 创建切割后的海陆分割保存文件夹
    lab_path = r"inputs/4096/new_label"
    if not os.path.exists(lab_path):
        os.makedirs(lab_path)

    """
    裁图
    block_size为结果图大小
    stride为裁图步长
    """
    cut_image(image_data, str(i), block_size=4096, stride=4096, save_path=train_img_path)
    cut_label(label_data, str(i), block_size=4096, stride=4096, save_path=lab_path)
