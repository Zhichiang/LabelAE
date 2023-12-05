import numpy as np
import torch
import torch.nn as nn


class MAE(nn.Module):
    """|outputs - targets|"""
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, outputs: torch.Tensor, target: torch.Tensor, *args):
        val_pixels = (target > 0).float()
        val_pixels = val_pixels.to(target.device)
        error = torch.abs(target * val_pixels - outputs * val_pixels)
        loss = torch.sum(error.view(error.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss/cnt)


class RMSE(nn.Module):
    """sqrt(pow(outputs - targets))"""
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, outputs: torch.Tensor, target: torch.Tensor, *args):
        val_pixels = (target > 0).float()
        val_pixels = val_pixels.to(target.device)
        error = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(error.view(error.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss/cnt))


class MRE(nn.Module):
    """|outputs - targets| / targets"""
    def __init__(self):
        super(MRE, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float()
        val_pixels = val_pixels.to(target.device)
        error = torch.abs(target * val_pixels - outputs * val_pixels)
        r = error / (target * val_pixels + 1e-6)
        r = r * val_pixels
        loss = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss/cnt)


class Deltas(nn.Module):
    def __init__(self):
        super(Deltas, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = (target > 0).float()
        val_pixels = val_pixels.to(target.device)
        rel = torch.max((target * val_pixels) / (outputs * val_pixels + 1e-6),
                        (outputs * val_pixels) / (target * val_pixels + 1e-6))
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)

        def del_i(i):
            r = (rel < 1.25 ** i).float() * val_pixels
            delta = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
            return torch.mean(delta)

        return del_i(1), del_i(2), del_i(3)


class IoURunningScore(object):
    label = ['road', 'sidewalk', 'building', 'wall',
             'fence', 'pole', 'light', 'sign',
             'vegetation', 'terrain', 'sky', 'person',
             'rider', 'car', 'truck', 'bus',
             'train', 'motorcycle', 'bicycle']

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    @ staticmethod
    def _fast_hist(label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) +
            label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        for id in range(19):
            print('===>' + self.label[id] + ':' + str(iu[id]))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc: \t': acc,
                'Mean Acc : \t': acc_cls,
                'FreqW Acc : \t': fwavacc,
                'Mean IoU : \t': mean_iu}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
