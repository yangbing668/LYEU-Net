import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from utils.averageMeter import AverageMeter


def init_log():
    log = {
        'loss': AverageMeter(),
        'time': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
        'acc': AverageMeter(),
        'precision': AverageMeter(),
        'recall': AverageMeter(),
        'f1': AverageMeter()
    }
    return log


def compute_dice2(pred, gt):
    pred = (pred >= .5).float()
    dice_score = (2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)
    return dice_score


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-8
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.mIoU = mIoULoss()

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        mIoUloss = self.mIoU(pred, target)
        loss = bceloss + diceloss + mIoUloss
        return loss


class DiceLoss1(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice_loss


class log_cosh_dice_loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(log_cosh_dice_loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        dice = compute_dice2(inputs, targets).item()
        loss = 1 - dice
        log_cosh = torch.log((torch.exp(loss) + torch.exp(-1 * loss)))
        return log_cosh


class TverskyLoss1(nn.Module):
    def __init__(self, alpha=1, beta=1):
        super(TverskyLoss1, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        fps = ((1 - targets) * preds).sum()
        fns = (targets * (1 - preds)).sum()
        tversky_score = intersection / (intersection + self.alpha * fns + self.beta * fps)
        return 1 - tversky_score


class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6, alpha=1, beta=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

        return 1 - Tversky


class mIoULoss(nn.Module):
    def __init__(self):
        super(mIoULoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: 预测的张量，形状通常为 (batch_size, 1, height, width) 等，值经过sigmoid等处理后在0到1之间表示属于目标类别的概率
        target: 对应的目标标签张量，形状通常与pred一致，元素取值为0或1
        """
        batch_size = pred.shape[0]
        # 将预测和目标张量的维度进行调整，将通道维度的1去掉，方便后续计算
        pred = pred.view(batch_size, -1)
        target = target.view(batch_size, -1)

        # 计算交集
        intersection = torch.sum(pred * target, dim=1)
        # 计算并集
        union = torch.sum(pred, dim=1) + torch.sum(target, dim=1) - intersection

        # 防止除零等数值问题，添加小的epsilon值
        epsilon = 1e-6
        # 计算IoU
        iou = (intersection + epsilon) / (union + epsilon)

        # 返回平均IoU损失（同样是1减去平均IoU）
        return 1 - torch.mean(iou)


def get_IoU(outputs, labels):
    EPS = 1e-6
    SR = outputs.view(-1)
    GT = labels.view(-1)
    # IoU of Foreground
    Inter1 = torch.sum((SR > 0.5) & (GT > 0.5))
    Union1 = torch.sum(SR > 0.5) + torch.sum(GT > 0.5) - Inter1
    IoU1 = float(Inter1) / (float(Union1) + EPS)

    # IoU of Background
    Inter2 = torch.sum((SR < 0.5) & (GT < 0.5))
    Union2 = torch.sum(SR < 0.5) + torch.sum(GT < 0.5) - Inter2
    IoU2 = float(Inter2) / (float(Union2) + EPS)

    mIoU = (IoU1 + IoU2) / 2

    return mIoU



def get_IoU1(outputs, labels):
    epsilon = 1e-7
    y_true = (labels > 0.5).int()
    y_pred = (outputs > 0.5).int()
    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    return tp / (tp + fp + fn + epsilon)


def accuracy(preds, label):
    preds = (preds > 0.5).int()
    label = (label > 0.5).int()
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc


def precision_recall_f1(preds, label):
    epsilon = 1e-7
    y_true = (label > 0.5).int()
    y_pred = (preds > 0.5).int()
    # tp = (y_true * y_pred).sum().to(torch.float32)
    # fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    # fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    tp = torch.sum((y_pred == 1) & (y_true == 1))
    fn = torch.sum((y_pred == 0) & (y_true == 1))
    fp = torch.sum((y_pred == 1) & (y_true == 0))

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    # 计算总样本数
    N = y_true.numel()
    tn = N - (tp + fp + fn)
    recall_neg = tn / (tn + fp + epsilon)
    balanced_accuracy = (recall + recall_neg) / 2
    g_mean = torch.sqrt(recall * recall_neg)
    return precision, recall, f1, balanced_accuracy, g_mean


def confusion_mat(preds, label):
    y_true = (label > 0.5).int()
    y_pred = (preds > 0.5).int()
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    return tp, tn, fp, fn


if __name__ == '__main__':
    mask = torch.ones(1, 1, 128, 128)
    mask[:, :, 50:100, 50:100] = 0
    print("\n\n", mask[0][0], "\n\n")
    print("*************************************")
    print(mask.shape)
    print(compute_dice2(mask, mask))
    print(compute_dice2(mask, torch.zeros(mask.shape)))

    loss = TverskyLoss()
    print(loss(mask, mask), loss(mask, torch.zeros(mask.shape)))

    print(get_IoU(mask, mask), get_IoU(mask, torch.zeros(mask.shape)))

    print(accuracy(mask, torch.zeros(mask.shape)))
