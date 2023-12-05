import torch
from config import cfg
from torch.utils.data import DataLoader
from dataloader.Cityscapes.CityscapesDataset import CityscapesDataSet

DATASET_NAME_MAP = {
    # 'nyuv2depth': NYUv2DepthDataset,
    'cityscapes': CityscapesDataSet,
}


class GenericDataloader:
    def __init__(self, name, split_set=('train', 'val', 'test'), max_iters=None):
        self.dataset_name = name
        self.loaders = {}
        datasets = [DATASET_NAME_MAP[name](split=v, max_iters=max_iters) for v in split_set]
        datasets_dict = dict(zip(split_set, datasets))

        if 'train' in split_set:
            self.loaders['train'] = DataLoader(datasets_dict['train'],
                                               batch_size=cfg.SOLVER.image_per_batch,
                                               shuffle=True, num_workers=6, drop_last=True, pin_memory=True)

        if 'val' in split_set:
            self.loaders['val'] = DataLoader(datasets_dict['val'],
                                             batch_size=cfg.SOLVER.val_image_per_batch,
                                             shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

        if 'test' in split_set:
            self.loaders['test'] = DataLoader(datasets_dict['test'],
                                              batch_size=cfg.TEST.image_per_batch,
                                              shuffle=False, num_workers=0, pin_memory=True)

