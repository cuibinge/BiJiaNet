import numpy as np  # 导入numpy库，用于处理数组和矩阵运算
from PIL import Image  # 导入PIL库的Image模块，用于处理图像
import os  # 导入os库，用于处理文件和目录操作


def load_images(img_folder,labels_folder):
    images = []  # 创建一个空列表，用于存储图像数据
    labels = []  # 创建一个空列表，用于存储标签数据
    for filename in os.listdir(img_folder):  # 遍历指定文件夹中的所有文件
        img = Image.open(os.path.join(img_folder, filename))  # 打开当前文件，并将其转换为Image对象
        # img = img.resize((512, 512))  # 将图像缩放到32x32大小
        img_array = np.array(img) / 512.0  # 将图像数据转换为numpy数组，并将像素值归一化到0-1之间
        images.append(img_array)  # 将处理后的图像数据添加到images列表中
        # labels.append(int(filename.split('_')[0]))  # 从文件名中提取标签信息，并将其添加到labels列表中
    for filename in os.listdir(labels_folder):
        label = Image.open(os.path.join(labels_folder, filename))
        # label = label.resize((512, 512))
        label_array = np.array(label)         #标签不进行归一化处理
        labels.append(label_array)
    return np.array(images), np.array(labels)


def save_as_npz(images, labels, output_file):
    for image,label in images,labels:
        np.savez(output_file, images=image, labels=label)  # 将images和labels数组保存为一个npz文件


img_folder = 'E:\\Datasets\\jiaozhouwan\\Data enhancement\\img\\image'               # 指定包含图像文件的文件夹路径
labels_folder = 'E:\\Datasets\\jiaozhouwan\\Data enhancement\\img\\01'
images, labels = load_images(img_folder,labels_folder)
output_file = 'E:\\NET\\project_TransUNet\\data\\Jiaozhouwan\\train_npz'       # 指定输出的npz文件路径
save_as_npz(images, labels, output_file)