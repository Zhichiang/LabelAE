import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gauss(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val += kernel[i, j]

    kernel = kernel / sum_val

    return kernel


class BoundaryGenerator(nn.Module):
    def __init__(self, ignore_index=None, kernel_size=1):
        super(BoundaryGenerator, self).__init__()
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index

        self.register_buffer('g_kernel', torch.Tensor(gauss(kernel_size, 2 / kernel_size))[None][None])
        b_kernel = torch.ones(1, 1, 3, 3)
        b_kernel[0, 0, 1, 1] = -8
        self.register_buffer('b_kernel', b_kernel)

    def forward(self, seg_map):
        seg_map = seg_map.clone().float()
        if len(seg_map.shape) == 3:
            seg_map = seg_map[None].transpose(0, 1)
        if self.ignore_index is not None:
            seg_map[seg_map == self.ignore_index] = 255

        with torch.no_grad():
            seg_map = F.conv2d(seg_map, self.b_kernel, padding=1)
            seg_map = torch.abs(seg_map)
            if self.ignore_index is not None:
                seg_map[seg_map >= 200] = 0
            seg_map[seg_map > 0] = 1
            seg_map = F.conv2d(seg_map, self.g_kernel, padding=self.kernel_size//2)
        return seg_map


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloader.TransferDataloader import TransferDataloader
    from config import cfg
    cfg.SOLVER.on_device = "162"
    loader = TransferDataloader('aaa')
    bgen = BoundaryGenerator(kernel_size=7)
    for data in loader.loaders['train']:
        img, seg = data
        seg_b = bgen(seg)
        plt.figure()
        plt.imshow(seg[0])
        plt.figure()
        plt.imshow(seg_b[0][0])
        pass
    pass
