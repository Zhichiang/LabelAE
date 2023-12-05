import argparse

from config import cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-file', type=str, help="config file",
                        default='config/cityscapes/cityscapes_sseg_labelae.yaml')
    parser.add_argument('--latent', type=int, help="latent code length", default=None)
    parser.add_argument('--slct', type=int, help="if select latent code", default=None)
    parser.add_argument('--slctlayer', type=int, help="if select layer", default=None)
    parser.add_argument('--epochs', type=int, help="max epochs", default=None)
    parser.add_argument('--maxiters', type=int, help="max iterations", default=None)
    parser.add_argument('--gpu', type=str, help="gpu id", default=None)
    parser.add_argument('--model', type=str, help="model name", default=None)
    parser.add_argument('--ondevice', type=str, help="device name", default=None)

    return parser.parse_args()


def merge_args(args):
    cfg.SOLVER.num_epochs = int(args.epochs) if args.epochs is not None else cfg.SOLVER.num_epochs
    cfg.SOLVER.max_iters = int(args.maxiters) if args.maxiters is not None else cfg.SOLVER.max_iters
    cfg.SOLVER.on_device = str(args.ondevice) if args.ondevice is not None else cfg.SOLVER.on_device
    cfg.MODEL.latent_len = int(args.latent) if args.latent is not None else cfg.MODEL.latent_len
    cfg.MODEL.select_latent = int(args.slct) if args.slct is not None else cfg.MODEL.select_latent
    cfg.MODEL.select_layer = int(args.slctlayer) if args.slctlayer is not None else cfg.MODEL.select_layer
    cfg.MODEL.gpu_id = args.gpu if args.gpu is not None else cfg.MODEL.gpu_id
    cfg.MODEL.name = args.model if args.model is not None else cfg.MODEL.name
