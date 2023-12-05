import torch
import torch.nn as nn
import torch.nn.functional as F


class CriterionDSN(nn.Module):
    """DSN : We need to consider two supervision for the model."""

    def __init__(self, ignore_index=255, use_weight=True, reduction='mean'):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction)
        if not reduction:
            print("disabled the reduction.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        if len(preds) >= 2:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss1 = self.criterion(scale_pred, target)

            scale_pred = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
            loss2 = self.criterion(scale_pred, target)
            return loss1 + loss2*0.4
        else:
            scale_pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
            loss = self.criterion(scale_pred, target)
            return loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()

    def forward(self, outputs: torch.Tensor, target: torch.Tensor, *args):
        val_pixels = torch.ne(target, 0).float()
        val_pixels = val_pixels.to(target.device)
        loss = F.smooth_l1_loss(outputs*val_pixels, target*val_pixels, reduction='none')
        loss_data = torch.sum(loss) / (1 + torch.sum(val_pixels))
        return loss_data


class CrossEntropy2dLoss(nn.Module):
    def __init__(self):
        super(CrossEntropy2dLoss, self).__init__()

    def forward(self, inputs, target, weight=None, size_average=True):
        n, c, h, w = inputs.size()
        log_p = F.log_softmax(inputs, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, ignore_index=255,
                          weight=weight, size_average=False)
        if size_average:
            loss /= mask.data.sum()
        return loss
