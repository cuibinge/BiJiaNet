# import imageio
import os
import numpy as np
import glob
from PIL import Image
from PIL import ImageFile
# import imageio.v2 as imageio
import imageio
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# 根目录为当前文件所在目录
root = os.getcwd()


# ====================================================================================================================================

def cut_image(image, image_raw, image_name, block_size, stride, save_path):
    row = image.shape[0]
    col = image.shape[1]
    dep = image.shape[2]
    # if row % block_size != 0 or col % block_size != 0:
    print('Need padding the image...')
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) * block_size
    padding_w = (col // block_size + 1) * block_size
    # 初始化一个 0 矩阵，将图像的值赋值到 0 矩阵的对应位置
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='uint8')
    padding_img[:row, :col, :] = image[:row, :col, :]

    row_num = 0
    for i in list(np.arange(0, row, stride)):
        row_num += 1

        if (i + block_size) > row:
            i_ = row - block_size
        else:
            i_ = i

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1

            if (j + block_size) > col:
                j_ = col - block_size
            else:
                j_ = j

            block = np.array(padding_img[i_: i_ + block_size, j_: j_ + block_size, :])
            block_name = image_name + '_' + str(int(row_num)) + '_' + str(int(col_num)) + '.tif'
            # print(block.shape)
            # print(block.min())
            # print(block.max())
            # block = block.astype(np.uint8)
#             if not np.all(image_raw[i_: i_ + block_size, j_: j_ + block_size, :] == 0, axis=2).any():
            if np.any(image_raw[i_: i_ + block_size, j_: j_ + block_size, :] != 0):
                imageio.imwrite(os.path.join(save_path, block_name), block)


def cut_label(image, image_raw, image_name, block_size, stride, save_path):
    row = image.shape[0]
    col = image.shape[1]

    # if row % block_size != 0 or col % block_size != 0:
    print('Need padding the image...')
    # 计算填充后图像的 hight 和 width
    padding_h = (row // block_size + 1) * block_size
    padding_w = (col // block_size + 1) * block_size
    # 初始化一个 0 矩阵，用于将预测结果的值赋值到 0 矩阵的对应位置
    padding_pre = np.zeros((padding_h, padding_w), dtype='uint8')
    padding_pre[:row, :col] = image[:row, :col]

    row_num = 0
    for i in list(np.arange(0, row, stride)):
        row_num += 1

        if (i + block_size) > row:
            i_ = row - block_size
        else:
            i_ = i

        col_num = 0
        for j in list(np.arange(0, col, stride)):
            col_num += 1

            if (j + block_size) > col:
                j_ = col - block_size
            else:
                j_ = j

            block = np.array(padding_pre[i_: i_ + block_size, j_: j_ + block_size])
            #            if np.min(block) == 0:
            #                print('ok')
            block_name = image_name + '_' + str(int(row_num)) + '_' + str(int(col_num)) + '.tif'
            # block[block == 255] = 0
            # print(block.shape)
            # print(block.min())
            # print(block.max())
            # block = block * 255
            # block = block.astype(np.uint8)
#             if not np.all(image_raw[i_: i_ + block_size, j_: j_ + block_size, :] == 0, axis=2).any():
            if np.any(image_raw[i_: i_ + block_size, j_: j_ + block_size, :] != 0):
                imageio.imwrite(os.path.join(save_path, block_name), block)


def truncate_stretch(image, min_percentile=2, max_percentile=98):
    stretched_channels = []
    # 排除所有通道全为0的像素
    image_n0 = image[np.any(image != 0, axis=-1)]
    for channel in range(image.shape[2]):
        image_channel = image[:, :, channel]

        # 计算每个通道的百分位数
        min_val = np.percentile(image_n0[:, channel], min_percentile)
        max_val = np.percentile(image_n0[:, channel], max_percentile)
#         print("min: ", min_val)
#         print("max: ", max_val)
        # 将像素值限制在截断范围内
        clipped_channel = np.clip(image_channel, min_val, max_val)

        # 缩放到0-255范围内
        stretched_channel = ((clipped_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        stretched_channels.append(stretched_channel)

    # 将各个通道重新堆叠成图像
    stretched_image = np.stack(stretched_channels, axis=2)

    return stretched_image


# ====================================================================================================================================

"""
生成彩色图像，为了标记样本
"""

# image 文件路径
image_path = r"/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/1"
image_path_list_raw = glob.glob(os.path.join(image_path, "*.tif")) + glob.glob(
    os.path.join(image_path, "*.png")) + glob.glob(os.path.join(image_path, "*.tiff"))

# # label 文件路径
# label_path = r"/HOME/scw6816/run/data/YZQ/LYG_DL/ann/val"
# label_path_list_raw = glob.glob(os.path.join(label_path, "*.tif")) + glob.glob(
#     os.path.join(label_path, "*.png")) + glob.glob(os.path.join(label_path, "*.tiff"))

image_path_list = sorted(image_path_list_raw)
# label_path_list = sorted(label_path_list_raw)

# mean_all = np.array([0, 0, 0, 0])
# std_all = np.array([0, 0, 0, 0])

# arr = np.array([0, 0, 0, 0, 0, 0])
# arr = np.zeros(25, dtype=np.int64)
for i in range(len(image_path_list)):
    # if i < 303:
    #     continue
    # 整幅图像的数据和标签
#     image_data_raw = imageio.imread(image_path_list[i])
    image_data = imageio.imread(image_path_list[i])
#     label_data = imageio.imread(label_path_list[i])

    img8_path = r"/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/test"
    if not os.path.exists(img8_path):
        os.makedirs(img8_path)

    file_name = os.path.basename(image_path_list[i])
    file_name = os.path.splitext(file_name)[0]
    print(file_name)
    print(image_data.shape)

    # 如果图像数据的形状为 (channels, height, width)，则转置为 (height, width, channels)
    if image_data.shape[0] == 3:  # 这里假设 channels 是第一个维度
        image_data = np.transpose(image_data, (1, 2, 0))

#     image_data = image_data[:, :, [2, 1, 0]]
#     print(image_data.shape)
    image_data = truncate_stretch(image_data)
    image_data = image_data[:, :, [2, 1, 0]]
    print(image_data.shape)

    imageio.imwrite(os.path.join(img8_path, file_name + '.png'), image_data)

    # # print(label_data.shape)
    # for j in range(25):
    #     arr[j] = arr[j] + (label_data == j).sum()
    #
    # print(arr)
    # image_data2 = image_data[54:6854, 60:7260]
    # mean = np.array([image_data2[:, :, 0].mean(), image_data2[:, :, 1].mean(), image_data2[:, :, 2].mean(),
    #                  image_data2[:, :, 3].mean()
    #                  ])
    # std = np.array([image_data2[:, :, 0].std(), image_data2[:, :, 1].std(), image_data2[:, :, 2].std(),
    #                 image_data2[:, :, 3].std()
    #                 ])
    # image_nan = np.where(np.all(image_data == 0, axis=-1, keepdims=True), np.nan, image_data)
    # mean = np.nanmean(image_nan, axis=(0, 1))
    # std = np.nanstd(image_nan, axis=(0, 1))
    # mean_all = mean_all + mean
    # std_all = std_all + std
    # print(mean)
    # print(std)
    # print(mean_all)
    # print(std_all)
    print("-------", i, "-------")
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

#     train_img_path = r"/HOME/scw6816/run/data/YZQ_cut/LYG_DL/img_dir/val"
# #     train_img_path = os.path.join(train_img_path, file_name)
#     if not os.path.exists(train_img_path):
#         os.makedirs(train_img_path)

#     lab_path = r"/HOME/scw6816/run/data/YZQ_cut/LYG_DL/ann_dir/val"
#     if not os.path.exists(lab_path):
#         os.makedirs(lab_path)

#     '''
#     裁图
#     block_size为结果图大小
#     stride为裁图步长
#     '''

# #     image_data=image_data[54:6854, 60:7260][:,:,[1,2,3]]
# #     label_data=label_data[54:6854, 60:7260]
#     image_data=image_data[:,:,[2,1,0]]
    
#     cut_image(image_data, image_data_raw, str(i), block_size=512, stride=256,
#               save_path=train_img_path)
#     cut_label(label_data, image_data_raw, str(i), block_size=512, stride=256,
#               save_path=lab_path)

# len_ = len(image_path_list)
# print(len_)
# print(mean_all)
# print(std_all)
# print(mean_all / len_)
# print(std_all / len_)
