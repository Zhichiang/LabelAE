import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F

from utils.boundary_gen import BoundaryGenerator
from config import cfg


class GTA5Dataset(Dataset):
    def __init__(self, split, root=None, max_iters=None, crop_size=None,
                 mean=(128, 128, 128), scale=None, mirror=None, ignore_label=255):
        if cfg.SOLVER.on_device == "182":
            self.root = "D:/Datasets/GTA5"
        elif cfg.SOLVER.on_device == "163":
            self.root = "/home/dengzq/datasets/GTA5"
        elif cfg.SOLVER.on_device == "162":
            self.root = "/media/data1/datasets/Generic/GTA5"
        else:
            raise NotImplementedError
        if 'train' in split:
            list_path = './dataloader/list/gta5/train_modified.txt'
            self.crop_size = (512, 512)
            self.scale = True
            self.is_mirror = True
        else:
            raise NotImplementedError
        self.crop_size = crop_size if crop_size is not None else self.crop_size
        self.scale = scale if scale is not None else self.scale
        self.ignore_label = ignore_label
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.is_mirror = mirror if mirror is not None else self.is_mirror
        self.b_gen = BoundaryGenerator(kernel_size=5)
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        for name in self.img_ids:
            img_file = os.path.join(self.root, "images/%s" % name)
            label_file = os.path.join(self.root, "labels/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        if self.scale:
            # f_scale = 0.7 + random.random() * 1.4
            f_scale = 0.5 + random.random() * 1.0
            fy, fx = int(1863 * f_scale), int(1024 * f_scale)
            image = image.resize((fy, fx), Image.BICUBIC)
            label = label.resize((fy, fx), Image.NEAREST)
        image = F.center_crop(image, self.crop_size)
        label = F.center_crop(label, self.crop_size)
        if self.is_mirror and random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.long)
        image -= self.mean

        # whether image is edge map
        if random.random() < cfg.SOLVER.edge_r:
            image = self.b_gen(torch.from_numpy(label)[None])
            image = np.array(torch.cat((image[0], image[0], image[0])).permute(1, 2, 0))

        image = image.transpose((2, 0, 1))
        label_copy = 255 * np.ones(label.shape, dtype=np.long)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        return image.copy(), label_copy.copy()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data.dataloader import DataLoader
    cfg.SOLVER.on_device = "162"
    cfg.SOLVER.edge_r = 1.0
    dataset = GTA5Dataset(split='train')
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for data in loader:
        iamage, laabel = data
        plt.imshow(iamage[0][0])
        pass
    pass

