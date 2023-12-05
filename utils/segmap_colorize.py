import numpy as np


cs_valid_colors = [[128,  64, 128],
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
cs_label_colours = dict(zip(range(19), cs_valid_colors))


def decode_segmap(img):
    n_classes = 19
    map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
    for idx in range(img.shape[0]):
        temp = img[idx, :, :]
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, n_classes):
            r[temp == l] = cs_label_colours[l][0]
            g[temp == l] = cs_label_colours[l][1]
            b[temp == l] = cs_label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        map[idx, :, :, :] = rgb
    return map


if __name__ == "__main__":
    pass
