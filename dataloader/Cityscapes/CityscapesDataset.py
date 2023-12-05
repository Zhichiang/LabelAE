import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F

from config import cfg


class CityscapesDataSet(Dataset):
    def __init__(self, split, root=None, max_iters=None, crop_size=None,
                 mean=(128, 128, 128), scale=None, mirror=None, ignore_label=255):
        if cfg.SOLVER.on_device == "182":
            self.root = "D:/Datasets/CityScapes"
        elif cfg.SOLVER.on_device == "163":
            self.root = "/media/data2/datasets/cityscapes"
        elif cfg.SOLVER.on_device == "162":
            self.root = "/media/data1/datasets/DepthSemantic/cityscapes"
        else:
            raise NotImplementedError
        if 'train' in split:
            list_path = './dataloader/list/cityscapes/train.lst'
            # self.crop_size = (769, 769)
            self.crop_size = (512, 512)
            self.scale = True
            self.is_mirror = True
        elif 'val' in split:
            list_path = './dataloader/list/cityscapes/val.lst'
            # self.crop_size = (1024, 2048)
            self.crop_size = (512, 1024)
            self.scale = False
            self.is_mirror = False
            max_iters = None
        elif 'test' in split:
            list_path = './dataloader/list/cityscapes/test.lst'
            self.crop_size = ()
            self.scale = False
            self.is_mirror = False
            max_iters = None
        else:
            raise NotImplementedError
        self.split = split
        self.crop_size = crop_size if crop_size is not None else self.crop_size
        self.scale = scale if scale is not None else self.scale
        self.ignore_label = ignore_label
        self.mean = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.is_mirror = mirror if mirror is not None else self.is_mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip().split() for i_id in open(list_path)]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for item in self.img_ids:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            img_file = os.path.join(self.root, image_path)
            label_file = os.path.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })
        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        print('{} images are loaded!'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.files)

    def id2trainId(self, label, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in self.id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        if self.scale:
            # f_scale = 0.7 + random.random() * 1.4
            f_scale = 0.5 + random.random() * 1.0
            fy, fx = int(image.size[0] * f_scale), int(image.size[1] * f_scale)
            image = image.resize((fy, fx), Image.BICUBIC)
            label = label.resize((fy, fx), Image.NEAREST)
        if 'val' in self.split:
            image = image.resize((1024, 512), Image.BICUBIC)
            label = label.resize((1024, 512), Image.NEAREST)
        image = F.center_crop(image, self.crop_size)
        label = F.center_crop(label, self.crop_size)
        if self.is_mirror and random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.long)
        image -= self.mean

        image = image.transpose((2, 0, 1))
        label = self.id2trainId(label)

        return image.copy(), label.copy()


# class CityscapesDataset(Dataset):
#     def __init__(self, root=None, max_iters=None, resize=None,
#                  mean=(128, 128, 128), transform=None, split='val'):
#         super(CityscapesDataset, self).__init__()
#         self.n_classes = 19
#         if self.root is not None:
#             self.root = root
#         else:
#             self.root = cfg.DATASETS.root
#         self.resize = resize
#         self.mean = mean
#         self.transform = transform
#         # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
#         self.img_ids = [i_id.strip() for i_id in open(img_list_path)]
#         self.lbl_ids = [i_id.strip() for i_id in open(label_list_path)]
#
#         if max_iters is not None:
#             self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
#             self.lbl_ids = self.lbl_ids * int(np.ceil(float(max_iters) / len(self.lbl_ids)))
#
#         self.files = []
#         self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
#                               19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
#                               26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
#         self.set = split
#         for img_name, lbl_name in zip(self.img_ids, self.lbl_ids):
#             img_file = os.path.join(self.root, "leftImg8bit/%s/%s" % (self.set, img_name))
#             lbl_file = os.path.join(self.root, "gtFine/%s/%s" % (self.set, lbl_name))
#             self.files.append({
#                 "img": img_file,
#                 "label": lbl_file,
#                 "name": img_name
#             })
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#         datafiles = self.files[index]
#
#         image = Image.open(datafiles["img"]).convert('RGB')
#         label = Image.open(datafiles["label"])
#
#         # resize
#         if self.resize is not None:
#             image = image.resize((self.resize[1], self.resize[0]), Image.BICUBIC)
#             label = label.resize((self.resize[1], self.resize[0]), Image.NEAREST)
#         # transform
#         if self.transform is not None:
#             image, label = self.transform(image, label)
#
#         image = np.asarray(image, np.float32)
#         label = np.asarray(label, np.long)
#
#         # re-assign labels to match the format of Cityscapes
#         label_copy = 255 * np.ones(label.shape, dtype=np.long)
#         for k, v in self.id_to_trainid.items():
#             label_copy[label == k] = v
#
#         size = image.shape
#         image = image[:, :, ::-1]  # change to BGR
#         image -= self.mean
#         image = image.transpose((2, 0, 1)) / 128.0
#
#         return image.copy(), label_copy.copy()
#
#     def decode_segmap(self, img):
#         map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
#         for idx in range(img.shape[0]):
#             temp = img[idx, :, :]
#             r = temp.copy()
#             g = temp.copy()
#             b = temp.copy()
#             for l in range(0, self.n_classes):
#                 r[temp == l] = label_colours[l][0]
#                 g[temp == l] = label_colours[l][1]
#                 b[temp == l] = label_colours[l][2]
#
#             rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
#             rgb[:, :, 0] = r / 255.0
#             rgb[:, :, 1] = g / 255.0
#             rgb[:, :, 2] = b / 255.0
#             map[idx, :, :, :] = rgb
#         return map


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader
    cfg.DATASETS.root = "D:\Datasets\CityScapes"
    dataset = CityscapesDataSet(split='train')
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    for data in loader:
        image, label = data
    pass
