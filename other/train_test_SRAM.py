# -*- coding = utf-8 -*-
'''
# @time:2023/4/8 10:57
# Author:DFTL
# @File:test.py
'''

import argparse
import imageio
import os
import cv2
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
import torch
from utils.TT_Dataset import MyDataset4,MyDataset3
from models.CMTFNet.CMTFNet import CMTFNet
import tifffile as tiff
import svgwrite
from PIL import Image
from io import BytesIO
import base64



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_name = ['Sea', 'Land']

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
            img, label, λ = data
            img, label, λ = img.to(device), label.to(device), λ.to(device)
            # 整除批次
            if img.shape[0] != args.batch:
                break
            output = model(img.float(),λ.float())
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
        if epoch == 0 or (epoch % 2 == 0 and epoch >= 80):
            weight_name = 'epoch_' + str(epoch) + '_loss_' + str('{:.6f}'.format(avg_loss)) + '.pt'
            torch.save(model.state_dict(), os.path.join(args.weights_path, weight_name))
            print('epoch: {} | loss: {:.6f} | Saving model... \n'.format(epoch, avg_loss))

def predict(model, data_loader, args):
    model.eval()
    save_dir = "C:/Users/zzc/Desktop/other/result/CMTF_SRAM/"
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在

    for batch in data_loader:
        images, filenames, λ = batch
        images = images.to(device)
        λ = λ.to(device)

        # 跳过不完整的批次（根据你的原有逻辑）
        if images.shape[0] != args.batch:
            break

        # 前向传播
        outputs = model(images.float(),λ.float())

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
            filename = filenames[i]
            save_path = os.path.join(save_dir, filename)

            # 保存结果
            tiff.imwrite(save_path, binary_array)

def predict_vector(model, data_loader, args):
    model.eval()
    save_dir = "C:/Users/zzc/Desktop/other/result/CMTF_SRAM_1024/"
    os.makedirs(save_dir, exist_ok=True)

    # 形态学处理参数配置
    kernel = np.ones((5, 5), np.uint8)  # 可配置化参数
    morph_open_iter = 2
    morph_close_iter = 4

    # SVG绘制参数
    stroke_config = {
        "color": "green",
        "width": 1.5,
        "chunk_size": 1000  # 处理超大轮廓的分段大小
    }

    for batch in data_loader:
        images, filenames, λ = batch
        images = images.to(device)
        λ = λ.to(device)

        if images.shape[0] != args.batch:
            break

        outputs = model(images.float(), λ.float())

        for i in range(outputs.shape[0]):
            # 生成二值分割结果
            output = outputs[i].squeeze(0)
            numpy_array = output.permute(1, 2, 0).cpu().detach().numpy()

            # Softmax处理
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return e_x / e_x.sum(axis=-1, keepdims=True)

            probability_array = softmax(numpy_array)
            binary_array = (probability_array[:, :, 1] > 0.5).astype(np.uint8)

            # ----------------- 矢量转换核心逻辑 -----------------
            # 1. 形态学处理
            opened = cv2.morphologyEx(binary_array, cv2.MORPH_OPEN, kernel, iterations=morph_open_iter)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=morph_close_iter)

            # 2. 轮廓提取
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                print(f"警告：{filenames[i]}未检测到轮廓")
                continue
            max_contour = max(contours, key=cv2.contourArea)
            raw_points = max_contour.squeeze()

            # 3. 读取对应原图（根据实际数据路径调整）
            img_path = os.path.join('C:/Users/zzc/Desktop/other/datasets/jiaozhouwan/1024/img2/', filenames[i])
            target_image = tiff.imread(img_path)

            # 4. 图像预处理
            # 数据类型转换
            if target_image.dtype == np.float32 or target_image.dtype == np.float64:
                target_image = (target_image * 255).astype(np.uint8)
            elif target_image.dtype == np.uint16:
                target_image = cv2.convertScaleAbs(target_image, alpha=(255.0 / 65535.0))
            else:
                target_image = target_image.astype(np.uint8)

            # 通道处理
            if len(target_image.shape) == 2:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB)
            elif target_image.shape[2] == 3:
                target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

            # 5. 生成SVG
            dwg = svgwrite.Drawing(
                size=(target_image.shape[1], target_image.shape[0]),
                profile='tiny'
            )

            # 添加背景图
            buffered = BytesIO()
            Image.fromarray(target_image).save(buffered, format="PNG")
            dwg.add(dwg.image(
                href=f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}",
                insert=(0, 0),
                size=tuple(target_image.shape[:2][::-1])  # (width, height)
            ))

            # 绘制矢量路径（内存优化版）
            path_data = []
            for x, y in raw_points:
                if not path_data:
                    path_data.append(f"M {x},{y}")
                else:
                    path_data.append(f"L {x},{y}")
            # path_data.append("Z")  # 闭合路径

            dwg.add(dwg.path(
                d=" ".join(path_data),
                fill="none",
                stroke=stroke_config["color"],
                stroke_width=stroke_config["width"]
            ))

            # 保存矢量文件
            svg_path = os.path.join(save_dir, f"vector_{os.path.splitext(filenames[i])[0]}.svg")
            dwg.saveas(svg_path, pretty=True)

            print(f"生成矢量文件：{svg_path}")


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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Controller')
    parser.add_argument('--mode', type=str, default='pre_train', help='pre_train/test/train/final_test')
    parser.add_argument('--image_path', type=str, default=r"./datasets/jiaozhouwan/128/img")
    parser.add_argument('--label_path', type=str, default=r"./datasets/jiaozhouwan/128/shp")
    parser.add_argument('--weights_path', type=str, default='./checkpoints/CMTFNet_SPAM',
                        help='the path saving weights')
    parser.add_argument('--result_path', type=str, default='./result/CMTFNet', help='the path saving result')
    parser.add_argument('--val_image_path', type=str, default=r'./datasets/jiaozhouwan/1024/img2', help='val_imageset path')
    parser.add_argument('--val_label_path', type=str, default=r'./datasets/jiaozhouwan/yuantu/shp/shp.tif', help='val_labelset path')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='train_epochs')
    args = parser.parse_args()

    criterion = CrossEntropyLoss()


    # 加载数据集
    mydataset_train = MyDataset3(args.image_path, args.label_path)
    data_loader = DataLoader(dataset=mydataset_train, batch_size=args.batch, shuffle=True, pin_memory=True)
    print("The images in Dataset: %d" % len(mydataset_train))

    model = CMTFNet(num_classes=2).to(device)

    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    if not os.path.exists(args.weights_path):
        os.makedirs(args.weights_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    '''优化器更新参数'''
    optimizer1 = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # train(model, data_loader, optimizer1, criterion, args)


    weights = os.listdir(args.weights_path)
    best_acc = 0
    mydataset_val = MyDataset4(args.val_image_path, args.label_path)
    data_loader_val = DataLoader(dataset=mydataset_val, batch_size=args.batch, shuffle=True, pin_memory=True)
    for w in weights:
        model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
        predict_vector(model,data_loader_val,args)
        sda