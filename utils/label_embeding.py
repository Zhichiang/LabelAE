import torch


def one_hot_embed_2d(label):
    label_tensor = label.view(-1, 1).clone()        # [B, H, W] -> [BxHxW, 1]
    label_tensor[label_tensor == 255] = 19
    onehot_tensor = torch.FloatTensor(label_tensor.shape[-2], 20).to(label.device)  # [BxHxW, 20]
    onehot_tensor.zero_()
    onehot_tensor.scatter_(1, label_tensor, 1)
    onehot_tensor = onehot_tensor.view(*label.shape, 20)  # [B, H, W, 20]
    onehot_tensor = onehot_tensor.permute((0, 3, 1, 2))

    return onehot_tensor


if __name__ == "__main__":
    import torch.nn.functional as F

    input_tensor = torch.randint(0, 3, (2, 3, 2))
    out_tensor = one_hot_embed_2d(input_tensor)
    rec_out = F.softmax(out_tensor, dim=1).data.max(1)[1].cpu().numpy()
    pass
