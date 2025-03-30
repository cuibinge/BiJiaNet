# -*- coding = utf-8 -*-
'''
# @time:2023/4/8 10:57
# Author:DFTL
# @File:test.py
'''

import argparse
import os
import imageio
import cv2
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import imageio.v2 as imageio

import time
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
import torch

from utils.TT_Dataset import MyDataset,MyDataset2
from models.CMTFNet.CMTFNet import CMTFNet
# from Models.model0603 import KRModel
from torchvision import transforms
from PIL import Image
import tifffile as tiff
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_name = ['Sea', 'Land']

import svgwrite
from PIL import Image
from io import BytesIO
import base64

# entities_dict = {name:i for i,name in enumerate(class_name)}
# relation_dict = {rel:i for i,rel in enumerate()}


def z_score_normal(image_data):
    B1, B2, B3 = cv2.split(image_data)
    B_mean = np.mean(B1)
    B_std = np.std(B1)
    B1_normalization = ((B1 - B_mean) / B_std).astype('float32')

    B_mean = np.mean(B2)
    B_std = np.std(B2)
    B2_normalization = ((B2 - B_mean) / B_std).astype('float32')

    B_mean = np.mean(B3)
    B_std = np.std(B3)
    B3_normalization = ((B3 - B_mean) / B_std).astype('float32')

    # B_mean = np.mean(B4)
    # B_std = np.std(B4)
    # B4_normalization = ((B4 - B_mean) / B_std).astype('float32')

    image_data = cv2.merge([B1_normalization, B2_normalization, B3_normalization])
    return image_data

def train(model, data_loader, optimizer1, criterion, args):
    model.train()
    total_epochs = args.epoch  # 总的训练 epoch 数
    for epoch in range(total_epochs):
        start_time = time.time()  # 记录开始时间
        print('======================epoch:{}/{}========================='.format(epoch, total_epochs))
        epoch_loss = 0
        i = 0
        for data in data_loader:
            i += 1
            img, label = data
            img, label = img.to(device), label.to(device)

            # 整除批次
            if img.shape[0] != args.batch:
                break

            output = model(img.float())
            optimizer1.zero_grad()
            loss1 = criterion(output, label.long())


            label_ = torch.tensor([0, 1]).to(device)
            label_ = label_.repeat(args.batch, 1).view(-1)

            loss = loss1

            epoch_loss += loss.item()  # 累加损失

            loss.backward()
            optimizer1.step()

            print('\r', 'step: ', i, ' loss: {:.6f}'.format(loss.item()), end='', flush=True)

        avg_loss = epoch_loss / len(data_loader)
        print('\n avg_loss:{:.6f} \n'.format(avg_loss))

        # 记录当前 epoch 的耗时
        end_time = time.time()
        epoch_time = end_time - start_time

        # 计算预计剩余时间
        remaining_epochs = total_epochs - (epoch + 1)
        estimated_remaining_time = remaining_epochs * epoch_time  # 预计剩余时间（秒）

        # 转换为小时、分钟、秒格式
        remaining_hours = estimated_remaining_time // 3600
        remaining_minutes = (estimated_remaining_time % 3600) // 60
        remaining_seconds = estimated_remaining_time % 60

        print('Estimated remaining time: {:02}:{:02}:{:02}'.format(int(remaining_hours), int(remaining_minutes),
                                                                   int(remaining_seconds)))

        # 保存模型
        if epoch %2 == 0 and epoch>=80:
            weight_name = 'epoch_' + str(epoch) + '_loss_' + str('{:.6f}'.format(avg_loss)) + '.pt'
            torch.save(model.state_dict(), os.path.join(args.weights_path, weight_name))
            print('epoch: {} | loss: {:.6f} | Saving model... \n'.format(epoch, avg_loss))

def predict2(model, data_loader, args):
    model.eval()
    save_dir = "C:/Users/zzc/Desktop/other/result/CMTFNet2/"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    for batch in data_loader:
        images, filenames = batch
        images = images.to(device)

        # 跳过不完整的批次（根据你的原有逻辑）
        if images.shape[0] != args.batch:
            break

        # 前向传播
        outputs = model(images.float())

        # 处理每个样本
        for i in range(outputs.shape[0]):
            # 获取输出并处理（参考你的原有逻辑）
            output = outputs[i].squeeze(0)  # 假设输出形状是 [C, H, W]
            numpy_array = output.permute(1, 2, 0).cpu().detach().numpy()

            # 应用 softmax 和阈值化
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return e_x / e_x.sum(axis=-1, keepdims=True)

            probability_array = softmax(numpy_array)
            binary_array = (probability_array[:, :, 1] > 0.5).astype(np.uint8)

            # 构建保存路径
            filename = filenames[i]  # 文件名（如 "72.tif"）
            save_path = os.path.join(save_dir, filename)

            # 保存结果
            tiff.imwrite(save_path, binary_array)

def predict_large_image(model, large_image_path, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 读取大图
    large_image = tiff.imread(large_image_path).astype(np.float32)
    original_height, original_width = large_image.shape[:2]

    # 创建保存结果的数组
    result_map = np.zeros((original_height, original_width), dtype=np.uint8)
    count_map = np.zeros((original_height, original_width), dtype=np.uint8)

    # 滑动窗口参数设置
    window_size = 128
    stride = 64  # 可以根据需要调整重叠步长

    # 创建滑动窗口数据集
    class SlideWindowDataset(Dataset):
        def __init__(self, img):
            self.img = img
            self.coords = []

            # 生成滑动窗口坐标
            for y in range(0, img.shape[0], stride):
                for x in range(0, img.shape[1], stride):
                    if y + window_size > img.shape[0]:
                        y = img.shape[0] - window_size
                    if x + window_size > img.shape[1]:
                        x = img.shape[1] - window_size
                    self.coords.append((y, x))

            # 处理右下方边缘
            if (img.shape[0] - window_size) % stride != 0:
                self.coords.append((img.shape[0] - window_size, img.shape[1] - window_size))

        def __len__(self):
            return len(self.coords)

        def __getitem__(self, idx):
            y, x = self.coords[idx]
            window = self.img[y:y + window_size, x:x + window_size]
            return torch.from_numpy(window), (y, x)


    # 自定义 collate 函数
    def custom_collate(batch):
        windows = [item[0] for item in batch]
        coords = [item[1] for item in batch]  # 保持坐标为元组列表
        windows = torch.stack(windows, dim=0)
        return windows, coords
    dataset = SlideWindowDataset(large_image)
    # 创建数据加载器（应用自定义 collate）
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=custom_collate
    )

    # 处理每个窗口
    with torch.no_grad():
        for batch in loader:
            windows, coords = batch
            # windows = windows.to(device).unsqueeze(1)  # 添加通道维度 [B, 1, H, W]
            windows = windows.permute([0,-1,1,2])
            # print(windows.shape)
            # 前向传播
            outputs = model(windows.float())

            # 处理每个窗口结果
            for i in range(outputs.shape[0]):
                output = outputs[i].squeeze(0)  # 假设输出形状是 [C, H, W]
                numpy_array = output.permute(1, 2, 0).cpu().numpy()

                # 应用softmax和阈值化（保持原有逻辑）
                def softmax(x):
                    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                    return e_x / e_x.sum(axis=-1, keepdims=True)

                probability_array = softmax(numpy_array)
                binary_array = (probability_array[:, :, 1] > 0.5).astype(np.uint8)

                # 获取当前窗口坐标
                y_start, x_start = coords[i]
                y_end = y_start + window_size
                x_end = x_start + window_size

                # 累加到结果图
                result_map[y_start:y_end, x_start:x_end] += binary_array
                count_map[y_start:y_end, x_start:x_end] += 1

    # 平均重叠区域
    result_map = np.where(count_map > 0, result_map / count_map, 0)
    final_result = (result_map > 0.5).astype(np.uint8)

    # 保存结果
    save_path = os.path.join("C:/Users/zzc/Desktop/other/result/CMTFNet/",
                             os.path.basename(large_image_path))
    tiff.imwrite(save_path, final_result)
    print("推理结束..........................................")

def predict_and_vectorize(model, image_path, args):
    """端到端执行预测并生成矢量覆盖图"""
    # ---------------------- 1. 大图预测 ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 读取大图
    large_image = tiff.imread(image_path).astype(np.float32)
    original_height, original_width = large_image.shape[:2]

    # 初始化结果数组
    result_map = np.zeros((original_height, original_width), dtype=np.uint8)
    count_map = np.zeros_like(result_map)

    # 滑动窗口参数
    window_size = 128
    stride = 64

    # 滑动窗口数据集
    class SlideWindowDataset(Dataset):
        def __init__(self, img):
            self.img = img
            self.coords = []

            # 生成滑动窗口坐标
            for y in range(0, img.shape[0], stride):
                for x in range(0, img.shape[1], stride):
                    if y + window_size > img.shape[0]:
                        y = img.shape[0] - window_size
                    if x + window_size > img.shape[1]:
                        x = img.shape[1] - window_size
                    self.coords.append((y, x))

            # 处理右下方边缘
            if (img.shape[0] - window_size) % stride != 0:
                self.coords.append((img.shape[0] - window_size, img.shape[1] - window_size))

        def __len__(self):
            return len(self.coords)

        def __getitem__(self, idx):
            y, x = self.coords[idx]
            window = self.img[y:y + window_size, x:x + window_size]
            return torch.from_numpy(window), (y, x)

    def custom_collate(batch):
        windows = [item[0] for item in batch]
        coords = [item[1] for item in batch]  # 保持坐标为元组列表
        windows = torch.stack(windows, dim=0)
        return windows, coords

    dataset = SlideWindowDataset(large_image)
    # 创建数据加载器（应用自定义 collate）
    loader = DataLoader(
        dataset,
        shuffle=False,
        collate_fn=custom_collate
    )
    # 预测处理
    with torch.no_grad():
        for batch in loader:
            windows, coords = batch
            # windows = windows.to(device).unsqueeze(1)
            windows = windows.permute([0, -1, 1, 2])
            outputs = model(windows.float())

            for i in range(outputs.shape[0]):
                output = outputs[i].squeeze(0)  # 假设输出形状是 [C, H, W]
                numpy_array = output.permute(1, 2, 0).cpu().numpy()

                # 应用softmax和阈值化（保持原有逻辑）
                def softmax(x):
                    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                    return e_x / e_x.sum(axis=-1, keepdims=True)

                probability_array = softmax(numpy_array)
                binary_array = (probability_array[:, :, 1] > 0.5).astype(np.uint8)

                # 获取当前窗口坐标
                y_start, x_start = coords[i]
                y_end = y_start + window_size
                x_end = x_start + window_size

                # 累加到结果图
                result_map[y_start:y_end, x_start:x_end] += binary_array
                count_map[y_start:y_end, x_start:x_end] += 1

    # 最终二值化结果
    final_result = (result_map / np.maximum(count_map, 1) > 0.5).astype(np.uint8)

    # ---------------------- 2. 边缘提取与矢量化 ----------------------
    # 直接使用预测结果，无需读取文件
    edge_map = final_result.astype(np.uint8) * 255

    # 查找轮廓（直接使用二值结果）
    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 轮廓简化
    approx_contours = []
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_contours.append(approx)

    # 处理原始图像用于叠加显示
    target_image = tiff.imread(image_path)
    if target_image.dtype == np.uint16:
        target_image = cv2.convertScaleAbs(target_image, alpha=(255.0 / 65535.0))
    elif target_image.dtype == np.float32:
        target_image = (target_image * 255).astype(np.uint8)
    if len(target_image.shape) == 2:
        target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)

    # 在 predict_and_vectorize 函数中添加以下代码
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)  # 关键修复：自动创建目录

    # 生成完整保存路径
    output_filename = os.path.basename(image_path).replace(".tif", ".svg")
    output_path = os.path.join(output_dir, output_filename)

    # 创建 SVG 画布
    dwg = svgwrite.Drawing(
        filename=output_path,
        size=(target_image.shape[1], target_image.shape[0]),
        profile='tiny'
    )

    # 添加背景图
    buffered = BytesIO()
    Image.fromarray(target_image).save(buffered, format="PNG")
    dwg.add(dwg.image(
        href=f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}",
        insert=(0, 0),
        size=target_image.shape[:2][::-1]
    ))

    # 添加矢量轮廓
    for cnt in approx_contours:
        if len(cnt) < 2: continue
        points = cnt.squeeze().tolist()
        points_sorted = sorted(points, key=lambda p: (p[1] // 5, p[0]))  # 按Y轴分组排序
        path_data = "M " + " L ".join(f"{x},{y}" for x, y in points_sorted)
        dwg.add(dwg.path(d=path_data, fill="none", stroke="lime", stroke_width=1))
    dwg.save()
    print(f"处理完成: {image_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--mode', type=str, default='pre_train', help='pre_train/test/train/final_test')
    parser.add_argument('--image_path', type=str, default=r"./datasets/jiaozhouwan/128/img")
    parser.add_argument('--label_path', type=str, default=r"./datasets/jiaozhouwan/128/shp")
    parser.add_argument('--weights_path', type=str, default='./checkpoints/CMTFNet',
                        help='the path saving weights')
    parser.add_argument('--result_path', type=str, default='./result/CMTFNet', help='the path saving result')
    parser.add_argument('--val_image_path', type=str, default=r'./datasets/jiaozhouwan/yuantu/img/img.tif', help='val_imageset path')
    parser.add_argument('--val_label_path', type=str, default=r'./datasets/jiaozhouwan/yuantu/shp/shp.tif', help='val_labelset path')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='train_epochs')
    args = parser.parse_args()

    criterion = CrossEntropyLoss()


    # 加载数据集
    mydataset = MyDataset(args.image_path, args.label_path)
    data_loader = DataLoader(dataset=mydataset, batch_size=args.batch, shuffle=True, pin_memory=True)
    print("The images in Dataset: %d" % len(mydataset))

    model = CMTFNet(num_classes=2).to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    if not os.path.exists(args.weights_path):
        os.makedirs(args.weights_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    '''优化器更新参数'''
    optimizer1 = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # 开始训练
    # train(model, data_loader, optimizer1, criterion, args)
    mydataset2 = MyDataset2(args.image_path)
    data_loader_predict = DataLoader(dataset=mydataset2, batch_size=args.batch, shuffle=True, pin_memory=True)
    # 验证
    label_data = imageio.imread(args.val_label_path) - 1
    image_data = imageio.imread(args.val_image_path)
    image_data = z_score_normal(image_data)
    weights = os.listdir(args.weights_path)
    best_acc = 0
    # for w in weights:
    #     model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
    #     predict2(model, data_loader_predict, args)


    # for w in weights:
    #     model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
    #     predict_large_image(model,args.val_image_path,args)

    for w in weights:
        model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
        predict_and_vectorize(model,args.val_image_path,args)
