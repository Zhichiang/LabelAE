import torch
from config import cfg
from torch.utils.data import DataLoader
from dataloader.Cityscapes.CityscapesDataset import CityscapesDataSet
from dataloader.GTA5.GTA5EdgeDataset import GTA5Dataset
from dataloader.transforms import *

DATASET_NAME_MAP = {
    # 'nyuv2depth': NYUv2DepthDataset,
    'cityscapes': CityscapesDataSet,
    'gta5': GTA5Dataset
}


class TransferDataloader:
    def __init__(self, name, split_set=('train', 'val', 'test'), max_iters=None):
        self.dataset_name = name
        self.loaders = {}

        if 'train' in split_set:
            self.loaders['train'] = DataLoader(GTA5Dataset(split='train', max_iters=max_iters),
                                               batch_size=cfg.SOLVER.image_per_batch,
                                               shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
            self.loaders['train_t'] = DataLoader(CityscapesDataSet(split='train', max_iters=max_iters),
                                                 batch_size=cfg.SOLVER.image_per_batch,
                                                 shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

        if 'val' in split_set:
            self.loaders['val'] = DataLoader(CityscapesDataSet(split='val', max_iters=max_iters),
                                             batch_size=cfg.SOLVER.val_image_per_batch,
                                             shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

