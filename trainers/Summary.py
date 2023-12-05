import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from utils.logger import setup_logger
from config import cfg


class Summary(object):
    def __init__(self, logs_dir='logs'):
        self.writer = SummaryWriter(log_dir=cfg.MODEL.logs_dir)
        self.logger = setup_logger(type(self).__name__)

    def write_data(self, scalars: dict, images: dict, batch_idx: int, epoch: int, per_epoch=1,
                   per_batch_s=1, per_batch_i=100, is_print=False, is_train=True, pre_name=None,
                   batch_per_epoch=None):
        if batch_per_epoch is not None:
            global_step = batch_idx + (epoch - 1) * batch_per_epoch
        else:
            if batch_idx == -1:
                global_step = epoch
            else:
                global_step = batch_idx

        if scalars is None:
            scalars = dict()
        if images is None:
            images = dict()

        if batch_idx % per_batch_s == 0 and epoch % per_epoch == 0:
            for tag in scalars:
                if pre_name is not None and '/' not in tag:
                    _tag = pre_name + '/' + tag
                else:
                    _tag = tag
                self.writer.add_scalar(_tag, scalars[tag], global_step)
        if batch_idx % per_batch_i == 0 and epoch % per_epoch == 0:
            for tag in images:
                if pre_name is not None and '/' not in tag:
                    _tag = pre_name + '/' + tag
                else:
                    _tag = tag
                image = make_grid(images[tag], padding=0, normalize=True, scale_each=True)
                self.writer.add_image(_tag, image, global_step)
        if is_print:
            phase = 'Train' if is_train else 'Val'
            for tag in scalars:
                info = '[{}] Epoch {}; Batch: {}; {}: {}'.format(phase, epoch, batch_idx, tag, scalars[tag])
                self.logger.info(info)
        pass

    def __del__(self):
        self.writer.close()


if __name__ == "__main__":
    pass
