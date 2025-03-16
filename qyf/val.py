import argparse
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from metrics import accuracy
from metrics import recall
from metrics import f1_score


from utils import AverageMeter
from albumentations import Compose, RandomRotate90, HorizontalFlip, VerticalFlip, OneOf, HueSaturationValue, RandomBrightnessContrast, Resize, Normalize
"""
需要指定参数：python val.py --name PNGData_SFFNet_woDS
"""

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='PNGData_SFFNet_woDS',
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        config['dataset'] = 'PNGData'
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.SFFNet().to(device)
    a = torch.rand(2, 3, 256, 256).to(device)
    # print(a.shape)
    res = model(a).to(device)
    print(res.shape)

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'test\\img', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    #
    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # 获取所有测试图片的文件路径
    img_ids = glob(os.path.join('inputs', config['dataset'], 'test\\img', '*' + config['img_ext']))
    # 只提取文件名（去除扩展名）
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # 直接使用所有图片作为验证集
    val_img_ids = img_ids

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        Resize(height=config['input_h'], width=config['input_w']),  # 调整图像大小
        Normalize(),  # 归一化
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'test\\img'),
        mask_dir=os.path.join('inputs', config['dataset'], 'test\\gtgt'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        #transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # from sklearn.metrics import f1_score, accuracy_score, recall_score
    # import numpy as np

    # Initialize AverageMeter for tracking IoU, F1, accuracy, and recall
    # 主要训练/验证循环
    avg_meter_iou = AverageMeter()
    avg_meter_accuracy = AverageMeter()
    avg_meter_recall = AverageMeter()
    avg_meter_f1 = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            # 计算IoU、准确率、召回率、F1-score
            iou = iou_score(output, target)
            accuracy_value = accuracy(output, target)
            recall_value = recall(output, target)
            f1_value = f1_score(output, target)

            avg_meter_iou.update(iou, input.size(0))
            avg_meter_accuracy.update(accuracy_value, input.size(0))
            avg_meter_recall.update(recall_value, input.size(0))
            avg_meter_f1.update(f1_value, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

        print('IoU: %.4f' % avg_meter_iou.avg)
        print('Accuracy: %.4f' % avg_meter_accuracy.avg)
        print('Recall: %.4f' % avg_meter_recall.avg)
        print('F1-Score: %.4f' % avg_meter_f1.avg)

    #iou评价指标
    # avg_meter = AverageMeter()
    #
    # for c in range(config['num_classes']):
    #     os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    # with torch.no_grad():
    #     for input, target, meta in tqdm(val_loader, total=len(val_loader)):
    #         input = input.cuda()
    #         target = target.cuda()
    #
    #         # compute output
    #         if config['deep_supervision']:
    #             output = model(input)[-1]
    #         else:
    #             output = model(input)
    #
    #         iou = iou_score(output, target)
    #         avg_meter.update(iou, input.size(0))
    #
    #         output = torch.sigmoid(output).cpu().numpy()
    #
    #         for i in range(len(output)):
    #             for c in range(config['num_classes']):
    #                 cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
    #                             (output[i, c] * 255).astype('uint8'))
    #
    # print('IoU: %.4f' % avg_meter.avg)
    
    plot_examples(input, target, model,num_examples=3)
    
    torch.cuda.empty_cache()

def plot_examples(datax, datay, model,num_examples=6):
    fig, ax = plt.subplots(nrows=num_examples, ncols=3, figsize=(18,4*num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx+1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0,:,:].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1,2,0))[:,:,0])
        ax[row_num][2].set_title("Target image")
    plt.show()




if __name__ == '__main__':
    main()
