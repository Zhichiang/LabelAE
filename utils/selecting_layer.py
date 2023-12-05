import torch
import torch.nn as nn


class SelectionLayer(nn.Module):
    def __init__(self, keep_layers=1, keep_percent=0.5, fix_layers=1, overlap=True):
        super(SelectionLayer, self).__init__()
        self.keep_layers = keep_layers
        self.keep_percent = keep_percent
        self.fix_layers = fix_layers
        self.overlap = overlap

    def forward(self, x):
        initial_shape = x.shape
        if self.overlap:
            feat_sum = initial_shape[1:].numel()
            max_sum = int(self.keep_percent * feat_sum)
        else:
            raise NotImplementedError
        max_mask = torch.zeros(x.shape).to(x.device)
        if self.fix_layers:
            max_mask[:, :self.fix_layers, :, :] = 1

        if self.keep_layers:
            max_v, max_i = x.topk(self.keep_layers, dim=1)
            max_mask.scatter_(1, max_i, 1)
        if self.overlap:
            x = x.view(initial_shape[0], -1)
            max_mask = max_mask.view(initial_shape[0], -1)
            max_v, max_i = x.topk(max_sum, dim=1)
            max_mask.scatter_(1, max_i, 1)

        min_mask = max_mask == 0
        x = x.masked_fill(min_mask, 0)
        x = x.reshape(initial_shape)
        return x


if __name__ == "__main__":
    select_layer = SelectionLayer()
    in_image = torch.randint(0, 10, (2, 3, 4, 4)).to(torch.float).requires_grad_(True)
    a_image = in_image * 2
    a_image = select_layer(a_image)
    y = a_image.sum()
    y.backward()
    pass
