import torch.nn as nn
# import torch.optim as optim

from modules.LabelAE.ReflectionAED import ReflectionEncoder, ReflectionDecoder
from modules.LabelAE.LabelAEv2 import LabelEncoder, LabelDecoder
from modules.SegNet.deeplabv3 import Seg_Model
from modules.LabelAE.ResNetEncoder import ResNetEncode
from modules.TransSegNet.TarDiscriminator import DomainDiscriminator

from utils.lr_scheduler import *
from utils.optimizer import *
from utils.logger import setup_logger
from config import cfg


def get_optimizers(models: [nn.Module]):
    logger = setup_logger("setup_parameters")

    optims = []
    lr_decays = []
    for model in models:
        # count_parameters
        params_count = 0
        params_training_count = 0
        for param in model.parameters():
            params_count += param.numel()
            if param.requires_grad:
                params_training_count += param.numel()
        logger.info('{} params count: {}'.format(type(model).__name__, params_count))
        logger.info('{} training params count: {}'.format(type(model).__name__, params_training_count))

        # set optimizers and lr_decays
        if "iscriminator" in type(model).__name__:
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = Adam(parameters, lr=1e-4,     # lr=cfg.SOLVER.base_lr,
                             weight_decay=cfg.SOLVER.weight_decay,
                             betas=(0.9, 0.99))
            lr_decay = globals()[cfg.SOLVER.lr_scheduler](optimizer,
                                                          step_size=cfg.SOLVER.lr_decay_step,
                                                          max_iter=cfg.SOLVER.max_iters,
                                                          gamma=cfg.SOLVER.lr_decay,
                                                          power=cfg.SOLVER.power,)
            optims.append(optimizer)
            lr_decays.append(lr_decay)
        else:
            parameters = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = globals()[cfg.SOLVER.optimizer](parameters, lr=cfg.SOLVER.base_lr,
                                                        weight_decay=cfg.SOLVER.weight_decay,
                                                        momentum=cfg.SOLVER.momentum)
            lr_decay = globals()[cfg.SOLVER.lr_scheduler](optimizer,
                                                          step_size=cfg.SOLVER.lr_decay_step,
                                                          max_iter=cfg.SOLVER.max_iters,
                                                          gamma=cfg.SOLVER.lr_decay,
                                                          power=cfg.SOLVER.power,)
            optims.append(optimizer)
            lr_decays.append(lr_decay)
    return optims, lr_decays


def build_model() -> [nn.Module]:
    if cfg.MODEL.name == 'multi':
        model_g = [ReflectionEncoder(19, cfg.MODEL.latent_len, cfg.MODEL.select_latent, cfg.MODEL.select_layer),
                   ReflectionDecoder(19, cfg.MODEL.latent_len, cfg.MODEL.select_latent, cfg.MODEL.select_layer)]
    elif cfg.MODEL.name == 'labelaev2':
        model_g = [LabelEncoder(19, cfg.MODEL.latent_len, cfg.MODEL.select_latent, cfg.MODEL.select_layer),
                   LabelDecoder(19, cfg.MODEL.latent_len, cfg.MODEL.select_latent, cfg.MODEL.select_layer)]
    elif cfg.MODEL.name == 'resnet_decoder':
        model_g = [ResNetEncode(pretrained=True),
                   LabelDecoder(19, cfg.MODEL.latent_len, cfg.MODEL.select_latent),
                   DomainDiscriminator()]
    elif cfg.MODEL.name == 'deeplabv3':
        model_g = [Seg_Model(19, pretrained_model="pretrained/resnet101-imagenet.pth")]
    else:
        raise NotImplementedError
    return model_g


if __name__ == "__main__":
    pass
