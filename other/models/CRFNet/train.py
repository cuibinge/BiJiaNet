# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:52:11 2022

@author: marti
"""

from IPython.display import clear_output
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.nn.init
from torch.autograd import Variable
from net.net import *
from utils.utils import *
from net.loss import *
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_dataset import *
from net.test_network import test
import cv2

import os
def train(net, optimizer, epochs, save_epoch, weights, train_loader, batch_size, window_size, output_folder, scheduler=None):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    criterion = nn.NLLLoss(weight=weights)
    iter_ = 0
    
    for e in tqdm(range(1, epochs + 1)):

        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target.data.cpu().numpy()
            target = np.transpose(target, [1,2,0])
            target3 = np.transpose(cv2.resize(target, dsize=(128, 128), interpolation=cv2.INTER_NEAREST), [2,0,1])
            target2 = np.transpose(cv2.resize(target, dsize=(64, 64), interpolation=cv2.INTER_NEAREST), [2,0,1])
            target1 = np.transpose(cv2.resize(target, dsize=(32, 32), interpolation=cv2.INTER_NEAREST), [2,0,1])
            target = np.transpose(target, [2,0,1])
            # add weights for mlp loss
            data, target = Variable(data.cuda()), Variable(torch.from_numpy(target).cuda())
            optimizer.zero_grad()
            output, out_fc, out_neigh = net(data)[:3]
            loss = CrossEntropy2d(output, target, weight=weights)
            loss_fc1 = CrossEntropy2d(out_fc[0], Variable(torch.from_numpy(target1).type(torch.LongTensor).cuda()), weight=compute_class_weight(target1).cuda())
            loss_fc2 = CrossEntropy2d(out_fc[1], Variable(torch.from_numpy(target2).type(torch.LongTensor).cuda()), weight=compute_class_weight(target2).cuda())
            loss_fc3 = CrossEntropy2d(out_fc[2], Variable(torch.from_numpy(target3).type(torch.LongTensor).cuda()), weight=compute_class_weight(target3).cuda())
            pairwise_loss = CrossEntropy2d(out_neigh, target, weight=weights)
            loss = (loss + loss_fc1 + loss_fc2 + loss_fc3) / 4 + pairwise_loss
            loss.backward()
            optimizer.step()

            #losses.i

            losses[iter_] = loss.item()  #loss.data[0]
            mean_losses[iter_] = np.mean(losses[max(0,iter_-100):iter_])
            
            if iter_ % 100 == 0:
                clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
                
                # 保存损失图像
                plt.figure()  # 创建一个新的图形窗口
                plt.plot(mean_losses[:iter_])
                plt.title('Mean Losses')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.savefig(os.path.join(output_folder, 'mean_losses_plot_iter_{}.png'.format(iter_)))  # 保存损失图像
                plt.close()  # 关闭图形窗口

                # 保存RGB、Ground truth和Prediction图像
                fig = plt.figure()

                ax1 = fig.add_subplot(131)
                ax1.imshow(rgb)
                ax1.set_title('RGB')
                ax1.axis('off')  # 隐藏坐标轴

                ax2 = fig.add_subplot(132)
                ax2.imshow(convert_to_color(gt))
                ax2.set_title('Ground truth')
                ax2.axis('off')

                ax3 = fig.add_subplot(133)
                ax3.imshow(convert_to_color(pred))
                ax3.set_title('Prediction')
                ax3.axis('off')

                plt.tight_layout()  # 自动调整布局
                fig.savefig(os.path.join(output_folder, 'segmentation_results','segmentation_results_iter_{}.png'.format(iter_)))  # 保存结果图像
                plt.close(fig)  # 关闭图形窗口

            iter_ += 1
            
            del(data, target, loss)

        if scheduler is not None:
            scheduler.step()

        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            torch.save(net.state_dict(), output_folder + '/test_epoch{}'.format(e))
    torch.save(net.state_dict(), output_folder + '/test_final')