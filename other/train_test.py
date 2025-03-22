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

from utils.TT_Dataset import MyDataset
from models.CMTFNet.CMTFNet import CMTFNet
# from Models.model0603 import KRModel
from torchvision import transforms
from PIL import Image
import tifffile as tiff
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_name = ['Sea', 'Land']


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


# ========================================================================================================================================



def model_predict_overlap_avg_prob(model, img_data, img_size, overlap=0.5, num_classes=2):
    model.eval()
    row, col, dep = img_data.shape
    stride = int(img_size * (1 - overlap))  # 计算步长，根据重叠率调整

    # 计算填充后的图像大小
    padding_h = ((row - 1) // stride + 1) * stride + img_size - stride
    padding_w = ((col - 1) // stride + 1) * stride + img_size - stride

    # 初始化填充图像、概率累加矩阵和计数矩阵
    padding_img = np.zeros((padding_h, padding_w, dep), dtype='float32')
    padding_img[:row, :col, :] = img_data[:row, :col, :]

    # 用于存储每个类别的概率累加值
    padding_prob_sum = np.zeros((padding_h, padding_w, num_classes), dtype='float32')
    count_map = np.zeros((padding_h, padding_w), dtype='float32')  # 用于记录每个像素被预测的次数

    # 对重叠区域的图像块进行预测
    for i in range(0, padding_h - img_size + 1, stride):
        for j in range(0, padding_w - img_size + 1, stride):
            # 取 img_size 大小的图像块
            img_data_ = padding_img[i:i + img_size, j:j + img_size, :]
            img_data_ = img_data_[np.newaxis, :, :, :]
            img_data_ = np.transpose(img_data_, (0, 3, 1, 2))
            img_data_ = torch.from_numpy(img_data_).to(device)

            # 模型预测，获取概率分布
            with torch.no_grad():
                y_pre = model(img_data_)
                y_prob = torch.squeeze(y_pre, dim=0).softmax(dim=0).cpu().numpy()
                y_prob = np.transpose(y_prob, (1, 2, 0))  # 调整形状为 (height, width, num_classes)

            # 将预测概率累加到结果矩阵中
            padding_prob_sum[i:i + img_size, j:j + img_size, :] += y_prob[:img_size, :img_size, :]
            count_map[i:i + img_size, j:j + img_size] += 1

    # 对概率矩阵进行归一化，计算平均概率
    avg_prob = padding_prob_sum / count_map[..., np.newaxis]

    # 根据平均概率选择具有最大值的类别作为最终预测
    padding_pre = np.argmax(avg_prob, axis=-1)

    # 返回裁剪到原始图像大小的预测结果
    return padding_pre[:row, :col].astype('uint8')


# ========================================================================================================================================

def calculation(y_label, y_pre, row, col):
    '''
    本函数主要计算以下评估标准的值：
    1、精准率
    2、召回率
    3、F1分数
    '''

    # 转成列向量
    y_label = np.reshape(y_label, (row * col, 1))
    y_pre = np.reshape(y_pre, (row * col, 1))

    y_label.astype('float64')
    y_pre.astype('float64')

    # 精准率
    precision = precision_score(y_label, y_pre, average=None)

    # 召回率
    recall = recall_score(y_label, y_pre, average=None)

    # F1
    f1 = f1_score(y_label, y_pre, average=None)

    # kappa
    kappa = cohen_kappa_score(y_label, y_pre)

    return precision, recall, f1, kappa


# ========================================================================================================================================

def estimate(y_label, y_pred, model_hdf5_name, class_name, dirname):
    '''
    本函数主要实现以下功能：
    1、计算准确率
    2、将各种评估指标存成一个json格式的txt文件
    @parameter:
        y_label:标签
        y_pred:预测结果
        model_hdf5_name:模型名
        class_name:类型
        dirname:存放路径
    '''

    # 准确率
    acc = np.mean(np.equal(y_label, y_pred) + 0)
    print('=================================================================================================')
    print('The estimate result of {} are as follows:'.format(model_hdf5_name))
    print('The acc of {} is {}'.format(model_hdf5_name, acc))

    precision, recall, f1, kappa = calculation(y_label, y_pred, y_label.shape[0], y_label.shape[1])

    # for i in range(len(class_name)):
    #     print('{}    F1: {:.5f}, Precision: {:.5f}, Recall: {:.5f}, kappa: {:.5f}'.format(class_name[i], f1[i],
    #                                                                                       precision[i], recall[i],
    #                                                                                       kappa))
    # print('=================================================================================================')
    if len(f1) == len(class_name):
        result = {}
        for i in range(len(class_name)):
            result[class_name[i]] = []
            tmp = {}
            tmp['Recall'] = str(round(recall[i], 5))
            tmp['Precision'] = str(round(precision[i], 5))
            tmp['F1'] = str(round(f1[i], 5))
            result[class_name[i]].append(tmp)

        result['Model Name'] = [model_hdf5_name]
        result['Accuracy'] = str(round(acc, 5))
        result['kappa'] = str(kappa)

        # 写入txt
        txt_name = "epoch_" + model_hdf5_name.split("_")[1] + "_acc_" + str(round(acc, 5))
        with open(os.path.join(dirname, txt_name + '.txt'), 'a', encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False))
    else:
        print("======================================>Estimate error!===========================================")
    return acc

def model_predict_single_patch(model, img_data, num_classes=2):
    """
    对单个小图块进行推理
    :param model: 已经加载的模型
    :param img_data: 输入的小图块数据，形状为 (height, width, depth)，类型为 float32
    :param num_classes: 分类的类别数，默认为2
    :return: 推理结果，形状为 (height, width)，类型为 uint8
    """
    model.eval()  # 设置模型为评估模式

    # 数据预处理：添加批次维度并转换为模型所需的格式
    img_data = img_data[np.newaxis, :, :, :]  # 添加批次维度
    img_data = np.transpose(img_data, (0, 3, 1, 2))  # 转换为 (batch_size, depth, height, width)
    img_data = img_data.to(device)  # 转换为张量并移动到设备上
    print(img_data.shape)
    # 模型预测
    with torch.no_grad():
        y_pre = model(img_data)
        y_prob = torch.squeeze(y_pre, dim=0).softmax(dim=0).cpu().numpy()  # 获取概率分布
        y_prob = np.transpose(y_prob, (1, 2, 0))  # 调整形状为 (height, width, num_classes)

    # 根据概率选择具有最大值的类别作为最终预测
    pre_label = np.argmax(y_prob, axis=-1).astype('uint8')

    return pre_label

def predict(model, data_loader, optimizer1, criterion, args):
    model.eval()
    total_epochs = 1  # 总的训练 epoch 数

    for epoch in range(total_epochs):
        start_time = time.time()  # 记录开始时间
        print('======================epoch:{}/{}========================='.format(epoch, total_epochs))
        for data in data_loader:
            img, _ = data
            img = img.to(device)
            # 整除批次
            if img.shape[0] != args.batch:
                break
            output = model(img.float())
            tensor = output.squeeze(0)
            numpy_array = tensor.detach().numpy()
            numpy_array = numpy_array.transpose(1, 2, 0)  # 转换形状为 (128, 128, 2)

            # 应用 softmax 归一化，将输出转换为概率
            # softmax 函数将每个像素的两个通道值转换为概率分布
            def softmax(x):
                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return e_x / e_x.sum(axis=-1, keepdims=True)

            probability_array = softmax(numpy_array)

            # 如果需要，可以将概率阈值化为二进制分类结果
            # 假设阈值为0.5，大于0.5的属于类别1，否则属于类别0
            binary_array = (probability_array[:, :, 1] > 0.5).astype(np.uint8)
            save_path = "C:/Users/zzc/Desktop/other/result/CMTFNet/1.tif"
            tiff.imwrite(save_path, binary_array)

            sdas





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


    # 验证
    label_data = imageio.imread(args.val_label_path) - 1
    image_data = imageio.imread(args.val_image_path)
    image_data = z_score_normal(image_data)
    weights = os.listdir(args.weights_path)
    best_acc = 0
    for w in weights:
        model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
        predict(model, data_loader, optimizer1, criterion, args)
    # for w in weights:
    #     image = imageio.imread('C:\\Users\\zzc\\Desktop\\other\\datasets\\jiaozhouwan\\128\\img\\64.tif')
    #     image = z_score_normal(image)
    #     image = torch.from_numpy(image)
    #     model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
    #     output = model_predict_single_patch(model,image,num_classes=2)
    #     save_name = "epoch_" + w.split("_")[1]  + ".tif"
    #     imageio.imwrite(os.path.join(args.result_path, save_name), output)
    #     print("Sucessfully saved to " + os.path.join(args.result_path, save_name))
    for w in weights:
        model.load_state_dict(torch.load(os.path.join(args.weights_path, w), map_location=device))
        output = model_predict_overlap_avg_prob(model, image_data, img_size=256, overlap=0.5, num_classes=2)
        # acc = estimate(label_data, output, w, class_name, args.result_path)
        # print('The acc of {} is {}'.format(w, acc))
        save_name = "epoch_" + w.split("_")[1] + ".tif"
        # save_name = "epoch_" + w.split("_")[1] + "_acc_" + ".tif"
        imageio.imwrite(os.path.join(args.result_path, save_name), output)
        # print("Sucessfully saved to " + os.path.join(args.result_path, save_name))
        # print('=================================================================================================')

