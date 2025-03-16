import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'CrossEntropyLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        #input = input.view(num, -1)
        input = input.reshape(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        """
        CrossEntropyLoss for multi-class classification.
        :param weight: Optional tensor of shape [C,] where C is the number of classes.
                       Used to re-weight the loss for each class.
        :param reduction: Reduction method, either 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        """
        Forward pass of the CrossEntropyLoss.
        :param input: Model output logits of shape [N, C] where N is the batch size and C is the number of classes.
        :param target: Ground truth labels of shape [N,] where N is the batch size.
        :return: Computed loss.
        """
        return self.loss_fn(input, target)