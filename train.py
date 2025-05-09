# import argparse
# import os
# from collections import OrderedDict
# from glob import glob
#
# import pandas as pd
# import torch
# import torch.backends.cudnn as cudnn
# import torch.nn as nn
# import torch.optim as optim
# import yaml
# from sklearn.model_selection import train_test_split
# from torch.optim import lr_scheduler
# from tqdm import tqdm
#
# import archs
# import SFFNet
# import losses
# from dataset import Dataset
# from metrics import iou_score
# from utils import AverageMeter, str2bool
# from albumentations import (
#     Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Resize, Normalize,
#     RandomBrightnessContrast, GaussNoise, GaussianBlur, OneOf, HueSaturationValue
# )
#
# ARCH_NAMES = archs.__all__
# LOSS_NAMES = losses.__all__
# LOSS_NAMES.append('BCEWithLogitsLoss')
#
# """
#
# 指定参数：
# --dataset PNGData  --arch SFFNet
# --dataset maweizao  --arch SFFNet
# --dataset train  --arch SFFNet
# --dataset PNGData  --arch NestedUNet
# """
#
#
# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('--name', default=None,
#                         help='model name: (default: arch+timestamp)')
#     parser.add_argument('--epochs', default=100, type=int, metavar='N',
#                         help='number of total epochs to run')
#     parser.add_argument('-b', '--batch_size', default=8, type=int,
#                         metavar='N', help='mini-batch size (default: 16)')
#
#     # model
#     parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
#                         choices=ARCH_NAMES,
#                         help='model architecture: ' +
#                              ' | '.join(ARCH_NAMES) +
#                              ' (default: SFFNet)')
#     parser.add_argument('--deep_supervision', default=False, type=str2bool)
#     parser.add_argument('--input_channels', default=4, type=int,
#                         help='input channels')
#     parser.add_argument('--num_classes', default=2, type=int,
#                         help='number of classes')
#     parser.add_argument('--input_w', default=256, type=int,
#                         help='image width')
#     parser.add_argument('--input_h', default=256, type=int,
#                         help='image height')
#
#     # loss
#     parser.add_argument('--loss', default='BCEDiceLoss',
#                         choices=LOSS_NAMES,
#                         help='loss: ' +
#                              ' | '.join(LOSS_NAMES) +
#                              ' (default: BCEDiceLoss)')
#
#     # dataset
#     parser.add_argument('--dataset', default='PNGData',
#                         help='dataset name')
#     parser.add_argument('--img_ext', default='.tif',
#                         help='image file extension')
#     parser.add_argument('--mask_ext', default='.tif',
#                         help='mask file extension')
#
#     # optimizer
#     parser.add_argument('--optimizer', default='Adam',
#                         choices=['Adam', 'SGD'],
#                         help='loss: ' +
#                              ' | '.join(['Adam', 'SGD']) +
#                              ' (default: Adam)')
#     parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
#                         metavar='LR', help='initial learning rate')
#     parser.add_argument('--momentum', default=0.9, type=float,
#                         help='momentum')
#     parser.add_argument('--weight_decay', default=1e-4, type=float,
#                         help='weight decay')
#     parser.add_argument('--nesterov', default=False, type=str2bool,
#                         help='nesterov')
#
#     # scheduler
#     parser.add_argument('--scheduler', default='CosineAnnealingLR',
#                         choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
#     parser.add_argument('--min_lr', default=1e-5, type=float,
#                         help='minimum learning rate')
#     parser.add_argument('--factor', default=0.1, type=float)
#     parser.add_argument('--patience', default=2, type=int)
#     parser.add_argument('--milestones', default='1,2', type=str)
#     parser.add_argument('--gamma', default=2 / 3, type=float)
#     parser.add_argument('--early_stopping', default=-1, type=int,
#                         metavar='N', help='early stopping (default: -1)')
#
#     parser.add_argument('--num_workers', default=0, type=int)
#
#     config = parser.parse_args()
#
#     return config
#
#
# def train(config, train_loader, model, criterion, optimizer):
#     avg_meters = {'loss': AverageMeter(),
#                   'iou': AverageMeter()}
#
#     model.train()
#
#     pbar = tqdm(total=len(train_loader))
#     for input, target, _ in train_loader:
#         input = input.cuda()
#         target = target.cuda()
#
#         # compute output
#         if config['deep_supervision']:
#             outputs = model(input)
#             loss = 0
#             for output in outputs:
#                 loss += criterion(output, target)
#             loss /= len(outputs)
#             iou = iou_score(outputs[-1], target)
#         else:
#             output = model(input)
#
#             loss = criterion(output, target)
#             iou = iou_score(output, target)
#
#         # compute gradient and do optimizing step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         avg_meters['loss'].update(loss.item(), input.size(0))
#         avg_meters['iou'].update(iou, input.size(0))
#
#         postfix = OrderedDict([
#             ('loss', avg_meters['loss'].avg),
#             ('iou', avg_meters['iou'].avg),
#         ])
#         pbar.set_postfix(postfix)
#         pbar.update(1)
#     pbar.close()
#
#     return OrderedDict([('loss', avg_meters['loss'].avg),
#                         ('iou', avg_meters['iou'].avg)])
#
#
# def validate(config, val_loader, model, criterion):
#     avg_meters = {'loss': AverageMeter(),
#                   'iou': AverageMeter()}
#
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         pbar = tqdm(total=len(val_loader))
#         for input, target, _ in val_loader:
#             input = input.cuda()
#             target = target.cuda()
#
#             # compute output
#             if config['deep_supervision']:
#                 outputs = model(input)
#                 loss = 0
#                 for output in outputs:
#                     loss += criterion(output, target)
#                 loss /= len(outputs)
#                 iou = iou_score(outputs[-1], target)
#             else:
#                 output = model(input)
#                 loss = criterion(output, target)
#                 iou = iou_score(output, target)
#
#             avg_meters['loss'].update(loss.item(), input.size(0))
#             avg_meters['iou'].update(iou, input.size(0))
#
#             postfix = OrderedDict([
#                 ('loss', avg_meters['loss'].avg),
#                 ('iou', avg_meters['iou'].avg),
#             ])
#             pbar.set_postfix(postfix)
#             pbar.update(1)
#         pbar.close()
#
#     return OrderedDict([('loss', avg_meters['loss'].avg),
#                         ('iou', avg_meters['iou'].avg)])
#
#
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     config = vars(parse_args())
#
#     if config['name'] is None:
#         if config['deep_supervision']:
#             config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
#         else:
#             config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
#     os.makedirs('models/%s' % config['name'], exist_ok=True)
#
#     print('-' * 20)
#     for key in config:
#         print('%s: %s' % (key, config[key]))
#     print('-' * 20)
#
#     with open('models/%s/config.yml' % config['name'], 'w') as f:
#         yaml.dump(config, f)
#
#     # define loss function (criterion)
#     if config['loss'] == 'BCEWithLogitsLoss':
#         criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
#     else:
#         criterion = losses.__dict__[config['loss']]().cuda()
#
#     cudnn.benchmark = True
#
#     # # create model unet++训练模板
#     # print("=> creating model %s" % config['arch'])
#     #
#     model = archs.__dict__[config['arch']](config['num_classes'],
#                                            config['input_channels'],
#                                            config['deep_supervision'])
#     # model=model.cuda()
#     x = torch.randn(8, 4, 256, 256)
#     num_channels = 4
#     start_wavelength = 450  # 起始波长 (nm)
#     end_wavelength = 900  # 终止波长 (nm)
#     batch_size = 8  # 批量大小
#
#     # 生成单个样本的波长向量
#     λ_single = torch.linspace(start_wavelength, end_wavelength, num_channels, dtype=torch.float32)  # [C]
#
#     # 将波长向量扩展到批量维度
#     λ_batch = λ_single.unsqueeze(0).expand(batch_size, -1)  # [B, C]
#
#     # 输出：tensor([450.0, 480.0, 510.0, ..., 900.0])
#     out = model(x, λ_batch)
#     print(out.shape)
#
#      # SFFNet训练模板
#     # model = archs.SFFNet().to(device)
#     # a = torch.rand(2, 4, 256, 256).to(device)
#     # # print(a.shape)
#     # res = model(a).to(device)
#     # print(res.shape)
#     # model = model.cuda()
#
#     # 创建模型并移动到设备
#
#     # model = archs.SFFNet().to(device)
#     # a = torch.rand(2, 4, 256, 256).to(device)
#
#     # # 前向传播并获取输出和特征图
#     # output, feature_maps = model(a)  # 现在模型返回两个值：输出和特征图字典
#     #
#     # # 打印输出形状
#     # print("Output shape:", output.shape)  # 应该是 [2, num_classes, 256, 256]
#     #
#     # # 打印所有可用的特征图键
#     # print("\nAvailable feature maps:")
#     # for key in feature_maps.keys():
#     #     print(f"- {key}")
#     #
#     # # 示例：查看特定特征图的形状
#     # print("\nExample feature map shapes:")
#     # print("Backbone res1 shape:", feature_maps['backbone_res1'].shape)
#     # print("Fuse feature L shape:", feature_maps['fusefeature_L'].shape)
#     # print("After WF1 shape:", feature_maps['after_WF1'].shape)
#     #
#     # # 将模型移动到CUDA（如果还没移动）
#     # model = model.cuda()
#     #
#     # # 如果你想可视化某个特征图（示例）
#     # import matplotlib.pyplot as plt
#     #
#     # def visualize_feature_map(feature_map, title="Feature Map"):
#     #     # 取第一个样本的第一个通道
#     #     sample = feature_map[0, 0].detach().cpu().numpy()
#     #     plt.figure(figsize=(10, 5))
#     #     plt.imshow(sample, cmap='viridis')
#     #     plt.title(title)
#     #     plt.colorbar()
#     #     plt.show()
#     #
#     # # 可视化某个特征图
#     # visualize_feature_map(feature_maps['backbone_res1'], "Backbone Res1 Feature Map")
#     # visualize_feature_map(feature_maps['final_output'], "Final Output")
#
#     params = filter(lambda p: p.requires_grad, model.parameters())
#     if config['optimizer'] == 'Adam':
#         optimizer = optim.Adam(
#             params, lr=config['lr'], weight_decay=config['weight_decay'])
#     elif config['optimizer'] == 'SGD':
#         optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
#                               nesterov=config['nesterov'], weight_decay=config['weight_decay'])
#     else:
#         raise NotImplementedError
#
#     if config['scheduler'] == 'CosineAnnealingLR':
#         scheduler = lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
#     elif config['scheduler'] == 'ReduceLROnPlateau':
#         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
#                                                    verbose=1, min_lr=config['min_lr'])
#     elif config['scheduler'] == 'MultiStepLR':
#         scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
#                                              gamma=config['gamma'])
#     elif config['scheduler'] == 'ConstantLR':
#         scheduler = None
#     else:
#         raise NotImplementedError
#
#     # Data loading code
#     # img_ids = glob(os.path.join('inputs', config['dataset'], 'train\\img', '*' + config['img_ext']))
#     img_ids = glob(os.path.join('inputs', config['dataset'], 'train\\image', '*' + config['img_ext']))
#     img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
#
#     # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
#     train_img_ids = img_ids
#     val_img_ids = glob(os.path.join('inputs', config['dataset'], 'test\\image', '*' + config['img_ext']))
#     val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
#     # 数据增强：
#
#     # 三通道训练集的数据增强
#     train_transform = Compose([
#         RandomRotate90(),  # 随机旋转 90 度
#         OneOf([
#             HorizontalFlip(p=0.5),  # 水平翻转
#             VerticalFlip(p=0.5),  # 垂直翻转
#         ], p=1),  # 按照归一化的概率选择执行哪一个
#         OneOf([
#             # HueSaturationValue(),  # 随机调整色调、饱和度和值
#             RandomBrightnessContrast(),  # 随机调整亮度和对比度
#         ], p=1),  # 按照归一化的概率选择执行哪一个
#         Resize(height=config['input_h'], width=config['input_w']),  # 调整图像大小
#         Normalize(),  # 归一化
#     ])
#
#     # 验证集的数据增强
#     val_transform = Compose([
#         Resize(height=config['input_h'], width=config['input_w']),  # 调整图像大小
#         Normalize(),  # 归一化
#     ])
#     # # 四通道训练集的数据增强
#     # train_transform = Compose([
#     #     RandomRotate90(),  # 随机旋转 90 度
#     #     OneOf([
#     #         HorizontalFlip(p=0.5),  # 水平翻转
#     #         VerticalFlip(p=0.5),  # 垂直翻转
#     #     ], p=1),  # 按照归一化的概率选择执行哪一个
#     #     OneOf([
#     #         RandomBrightnessContrast(p=0.5),  # 随机调整亮度和对比度
#     #         GaussNoise(p=0.5),  # 添加高斯噪声
#     #         GaussianBlur(p=0.5),  # 高斯模糊
#     #     ], p=1),  # 按照归一化的概率选择执行哪一个
#     #     Resize(height=config['input_h'], width=config['input_w']),  # 调整图像大小
#     #     Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),  # 归一化
#     # ])
#
#     # 验证集的数据增强
#     val_transform = Compose([
#         Resize(height=config['input_h'], width=config['input_w']),  # 调整图像大小
#         Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),  # 归一化
#     ])
#
#     train_dataset = Dataset(
#         img_ids=train_img_ids,
#         #
#         # img_dir=os.path.join('inputs',config['dataset'], 'train\\img'),
#         img_dir=os.path.join('inputs', config['dataset'], 'train\\image'),
#         # 'inputs',
#         # mask_dir=os.path.join('inputs',config['dataset'], 'train\\our_mask'),
#         mask_dir=os.path.join('inputs', config['dataset'], 'train\\label'),
#         img_ext=config['img_ext'],
#         mask_ext=config['mask_ext'],
#         num_classes=config['num_classes'],
#         # transform=train_transform
#     )
#
#     print(f"Image directory: {train_dataset.img_dir}")
#     print(f"Mask directory: {train_dataset.mask_dir}")
#
#     val_dataset = Dataset(
#         img_ids=val_img_ids,
#         img_dir=os.path.join('inputs', config['dataset'], 'test\\image'),
#         # img_dir=os.path.join('inputs', config['dataset'], 'image'),
#         # 'inputs',
#         mask_dir=os.path.join('inputs', config['dataset'], 'test\\label'),
#         # mask_dir=os.path.join('inputs', config['dataset'], 'label'),
#         img_ext=config['img_ext'],
#         mask_ext=config['mask_ext'],
#         num_classes=config['num_classes'],
#         # transform=val_transform
#     )
#
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=config['num_workers'],
#         drop_last=True)  # 不能整除的batch是否就不要了
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset,
#         # batch_size=config['batch_size'],
#         batch_size=1,
#         shuffle=False,
#         num_workers=config['num_workers'],
#         drop_last=False)
#
#     log = OrderedDict([
#         ('epoch', []),
#         ('lr', []),
#         ('loss', []),
#         ('iou', []),
#         ('val_loss', []),
#         ('val_iou', []),
#     ])
#
#     best_iou = 0
#     trigger = 0
#     for epoch in range(config['epochs']):
#         print('Epoch [%d/%d]' % (epoch, config['epochs']))
#
#         # train for one epoch
#         train_log = train(config, train_loader, model, criterion, optimizer)
#         # evaluate on validation set
#         val_log = validate(config, val_loader, model, criterion)
#
#         if config['scheduler'] == 'CosineAnnealingLR':
#             scheduler.step()
#         elif config['scheduler'] == 'ReduceLROnPlateau':
#             scheduler.step(val_log['loss'])
#
#         print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
#               % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
#
#         log['epoch'].append(epoch)
#         log['lr'].append(config['lr'])
#         log['loss'].append(train_log['loss'])
#         log['iou'].append(train_log['iou'])
#         log['val_loss'].append(val_log['loss'])
#         log['val_iou'].append(val_log['iou'])
#
#         pd.DataFrame(log).to_csv('models/%s/log.csv' %
#                                  config['name'], index=False)
#
#         trigger += 1
#
#         if val_log['iou'] > best_iou:
#             torch.save(model.state_dict(), 'models/%s/model.pth' %
#                        config['name'])
#             best_iou = val_log['iou']
#             print("=> saved best model")
#             trigger = 0
#
#         # early stopping
#         if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
#             print("=> early stopping")
#             break
#
#         torch.cuda.empty_cache()
#
#
# if __name__ == '__main__':
#     main()
import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs
import SFFNet
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from albumentations import (
    Compose, RandomRotate90, HorizontalFlip, VerticalFlip, Resize, Normalize,
    RandomBrightnessContrast, GaussNoise, GaussianBlur, OneOf, HueSaturationValue
)

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

"""

指定参数：
--dataset PNGData  --arch SFFNet
--dataset maweizao  --arch SFFNet
--dataset train  --arch SFFNet
--dataset PNGData  --arch NestedUNet
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: SFFNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=4, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=4096, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=4096, type=int,
                        help='image height')

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')

    # dataset
    parser.add_argument('--dataset', default='PNGData',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.tif',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.tif',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                             ' | '.join(['Adam', 'SGD']) +
                             ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, metadata in train_loader:
        # 确保所有张量都在同一设备上
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        λ = metadata['wavelengths'].float().cuda(non_blocking=True)  # 确保类型和设备正确

        # 调试检查
        # print(f"Input device: {input.device}")
        # print(f"Target device: {target.device}")
        # print(f"Wavelength device: {λ.device}")
        # print(f"Model device: {next(model.parameters()).device}")

        # compute output
        output = model(input, λ)
        loss = criterion(output, target)
        iou = iou_score(output, target)


        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                      ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                 'iou': AverageMeter()}

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, metadata in val_loader:
            input = input.cuda()
            target = target.cuda()
            λ = metadata['wavelengths'].float().cuda()  # 确保转换为float并转移到GPU

            # compute output
            if config['deep_supervision']:
                outputs = model(input, λ)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input, λ)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()  # WithLogits 就是先将输出结果经过sigmoid再交叉熵
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    # # create model unet++训练模板
    # print("=> creating model %s" % config['arch'])
    #
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    model=model.cuda()
    x = torch.randn(2, 4, 512, 512)
    num_channels = 4
    start_wavelength = 450  # 起始波长 (nm)
    end_wavelength = 900  # 终止波长 (nm)
    batch_size = 2  # 批量大小

    # 生成单个样本的波长向量
    # 生成单个样本的波长向量（直接在GPU上创建）
    λ_single = torch.linspace(
        start_wavelength,
        end_wavelength,
        num_channels,
        dtype=torch.float32,
        device='cuda'  # 直接在GPU上创建张量
    )  # [C]

    # 将波长向量扩展到批量维度（保持GPU设备）
    λ_batch = λ_single.unsqueeze(0).expand(batch_size, -1)  # [B, C]

    # 确保输入数据x也在GPU上
    x = x.cuda()  # 如果x尚未在GPU上

    # 模型计算（所有张量均在GPU上）
    out = model(x, λ_batch)
    out = model(x, λ_batch)
    print(out.shape)

     # SFFNet训练模板
    # model = archs.SFFNet().to(device)
    # a = torch.rand(2, 4, 256, 256).to(device)
    # # print(a.shape)
    # res = model(a).to(device)
    # print(res.shape)
    # model = model.cuda()

    # 创建模型并移动到设备

    # model = archs.SFFNet().to(device)
    # a = torch.rand(2, 4, 256, 256).to(device)

    # # 前向传播并获取输出和特征图
    # output, feature_maps = model(a)  # 现在模型返回两个值：输出和特征图字典
    #
    # # 打印输出形状
    # print("Output shape:", output.shape)  # 应该是 [2, num_classes, 256, 256]
    #
    # # 打印所有可用的特征图键
    # print("\nAvailable feature maps:")
    # for key in feature_maps.keys():
    #     print(f"- {key}")
    #
    # # 示例：查看特定特征图的形状
    # print("\nExample feature map shapes:")
    # print("Backbone res1 shape:", feature_maps['backbone_res1'].shape)
    # print("Fuse feature L shape:", feature_maps['fusefeature_L'].shape)
    # print("After WF1 shape:", feature_maps['after_WF1'].shape)
    #
    # # 将模型移动到CUDA（如果还没移动）
    # model = model.cuda()
    #
    # # 如果你想可视化某个特征图（示例）
    # import matplotlib.pyplot as plt
    #
    # def visualize_feature_map(feature_map, title="Feature Map"):
    #     # 取第一个样本的第一个通道
    #     sample = feature_map[0, 0].detach().cpu().numpy()
    #     plt.figure(figsize=(10, 5))
    #     plt.imshow(sample, cmap='viridis')
    #     plt.title(title)
    #     plt.colorbar()
    #     plt.show()
    #
    # # 可视化某个特征图
    # visualize_feature_map(feature_maps['backbone_res1'], "Backbone Res1 Feature Map")
    # visualize_feature_map(feature_maps['final_output'], "Final Output")

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'train\\img', '*' + config['img_ext']))
    img_ids = glob(os.path.join('inputs', config['dataset'], 'train\\image', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    train_img_ids = img_ids
    val_img_ids = glob(os.path.join('inputs', config['dataset'], 'test\\image', '*' + config['img_ext']))
    val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]


    train_dataset = Dataset(
        img_ids=train_img_ids,
        #
        # img_dir=os.path.join('inputs',config['dataset'], 'train\\img'),
        img_dir=os.path.join('inputs', config['dataset'], 'train\\image'),
        # 'inputs',
        # mask_dir=os.path.join('inputs',config['dataset'], 'train\\our_mask'),
        mask_dir=os.path.join('inputs', config['dataset'], 'train\\label'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        # transform=train_transform
    )

    print(f"Image directory: {train_dataset.img_dir}")
    print(f"Mask directory: {train_dataset.mask_dir}")

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'test\\image'),
        # img_dir=os.path.join('inputs', config['dataset'], 'image'),
        # 'inputs',
        mask_dir=os.path.join('inputs', config['dataset'], 'test\\label'),
        # mask_dir=os.path.join('inputs', config['dataset'], 'label'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        # transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)  # 不能整除的batch是否就不要了
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # batch_size=config['batch_size'],
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
