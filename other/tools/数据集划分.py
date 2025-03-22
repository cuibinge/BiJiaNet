import os
import shutil
import random


def create_dirs(paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


def split_data(img_dir, label_dir, train_img_dir, test_img_dir, train_label_dir, test_label_dir, split_ratio=0.8):
    # 获取所有的图片文件名
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.tif')]

    # 打乱文件顺序
    random.shuffle(img_files)

    # 计算训练集和测试集的分割点
    split_point = int(len(img_files) * split_ratio)

    train_files = img_files[:split_point]
    test_files = img_files[split_point:]

    # 确保目标路径存在
    create_dirs([train_img_dir, test_img_dir, train_label_dir, test_label_dir])

    # 复制文件到训练集和测试集
    for file in train_files:
        shutil.copy2(os.path.join(img_dir, file), os.path.join(train_img_dir, file))
        label_file = file.replace('.tif', '.tif')  # 假定标签文件扩展名也是 .tif
        shutil.copy2(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

    for file in test_files:
        shutil.copy2(os.path.join(img_dir, file), os.path.join(test_img_dir, file))
        label_file = file.replace('.tif', '.tif')  # 假定标签文件扩展名也是 .tif
        shutil.copy2(os.path.join(label_dir, label_file), os.path.join(test_label_dir, label_file))

    print(f"训练集包含 {len(train_files)} 个样本，测试集包含 {len(test_files)} 个样本。")


if __name__ == "__main__":
    # 总图像
    img_dir = r"C:\Users\zzc\Desktop\data\Data enhancement\img\image"
    # 总标签
    label_dir = r"C:\Users\zzc\Desktop\data\Data enhancement\class\image"
    # 训练图像
    train_img_dir = r"C:\Users\zzc\Desktop\data\Datasets\train\img"
    # 测试图像
    test_img_dir = r"C:\Users\zzc\Desktop\data\Datasets\test\img"
    # 训练标签
    train_label_dir = r"C:\Users\zzc\Desktop\data\Datasets\train\class"
    # 测试标签
    test_label_dir = r"C:\Users\zzc\Desktop\data\Datasets\test\class"

    split_data(img_dir, label_dir, train_img_dir, test_img_dir, train_label_dir, test_label_dir)