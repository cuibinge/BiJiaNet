import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def accuracy(output, target):
    """
    计算准确率（Accuracy）
    """
    output = (output > 0.5).float()  # 将布尔张量转换为浮动类型
    target = target.float()  # 确保 target 是浮动类型
    correct = torch.sum(output == target).float()
    total = target.numel()
    return correct / total

def recall(output, target):
    """
    计算召回率（Recall）
    """
    output = (output > 0.5).float()
    target = target.float()
    true_positive = torch.sum(output * target)
    false_negative = torch.sum((1 - output) * target)  # 注意，这里转换了布尔操作
    recall_value = true_positive / (true_positive + false_negative + 1e-6)  # 防止除以0
    return recall_value

def f1_score(output, target):
    """
    计算F1-score
    """
    precision = accuracy(output, target)
    recall_value = recall(output, target)
    f1 = 2 * (precision * recall_value) / (precision + recall_value + 1e-6)  # 防止除以0
    return f1
