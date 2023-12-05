import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


valid_colors = [[128,  64, 128],
                [244,  35, 232],
                [ 70,  70,  70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170,  30],
                [220, 220,   0],
                [107, 142,  35],
                [152, 251, 152],
                [ 70, 130, 180],
                [220,  20,  60],
                [255,   0,   0],
                [  0,   0, 142],
                [  0,   0,  70],
                [  0,  60, 100],
                [  0,  80, 100],
                [  0,   0, 230],
                [119,  11,  32]]
label_colours = dict(zip(range(19), valid_colors))


class GTA5Dataset(Dataset):
    def __init__(self, root, list_path, max_iters, resize=None, mean=(128, 128, 128), transform=None):
        super(GTA5Dataset, self).__init__()

        self.n_classes = 19
        self.root = root
        self.list_path = list_path
        self.resize = resize
        self.mean = mean
        self.transform = transform
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])

        # resize
        if self.resize is not None:
            image_pil = image.resize((self.resize[1], self.resize[0]), Image.BICUBIC)
            label_pil = label.resize((self.resize[1], self.resize[0]), Image.NEAREST)
        i_iter = 0
        while True:
            i_iter = i_iter + 1
            if i_iter > 5:
                print(datafiles["img"])
                break
            # transform
            if self.transform is not None:
                image, label = self.transform(image_pil, label_pil)

            image = np.asarray(image, np.float32)
            label = np.asarray(label, np.long)

            # re-assign labels to match the format of Cityscapes
            label_copy = 255 * np.ones(label.shape, dtype=np.long)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v

            label_cat, label_time = np.unique(label_copy, return_counts=True)
            label_p = 1.0 * label_time / np.sum(label_time)
            pass_c, pass_t = np.unique(label_p > 0.02, return_counts=True)
            if pass_c[-1]:
                if pass_t[-1] >= 3:
                    break
                elif pass_t[-1] == 2:
                    if not (label_cat[-1] == 255 and label_p[-1] > 0.02):
                        break

        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1)) / 128.0

        return image.copy(), label_copy.copy()

    def decode_segmap(self, img):
        segmap = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
        for idx in range(img.shape[0]):
            temp = img[idx, :, :]
            r = temp.copy()
            g = temp.copy()
            b = temp.copy()
            for l in range(0, self.n_classes):
                r[temp == l] = label_colours[l][0]
                g[temp == l] = label_colours[l][1]
                b[temp == l] = label_colours[l][2]

            rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
            rgb[:, :, 0] = r / 255.0
            rgb[:, :, 1] = g / 255.0
            rgb[:, :, 2] = b / 255.0
            segmap[idx, :, :, :] = rgb
        return segmap


if __name__ == "__main__":
    pass
