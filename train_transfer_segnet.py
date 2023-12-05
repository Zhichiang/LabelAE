import os
import pprint
import torch
import torch.nn as nn

from trainers.SegnetTransTrainerEnDe import SegnetTrainer
from dataloader.GenericDataloader import GenericDataloader
from dataloader.TransferDataloader import TransferDataloader

from utils.workspace_assign import name_assign
from utils.build_component import get_optimizers, build_model
from utils.args_parser import get_args, merge_args

from config import cfg
from config.defaults import merge_cfg_from_file
from utils.logger import setup_logger, LoggerSetup


def main():
    # Config reading and merge
    args = get_args()
    cfg_file = args.cfg_file
    merge_cfg_from_file(cfg_file)
    merge_args(args)

    train_name = name_assign()
    # Config the output directory
    os.makedirs(cfg.MODEL.workspace, exist_ok=True)
    cfg.MODEL.chkpt_dir = os.path.join(cfg.MODEL.workspace, "chkpt_"+train_name)
    cfg.MODEL.logs_dir = os.path.join(cfg.MODEL.workspace, 'logs', train_name)

    # Set up logger
    log_file_name = os.path.join(cfg.MODEL.workspace, train_name)
    LoggerSetup.file_name = log_file_name
    logger = setup_logger("depth_trainer_logger")

    logger.info(pprint.pformat(cfg))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.gpu_id

    # prepare dataloader
    dataset = TransferDataloader(cfg.DATASETS.name, split_set=('train', 'val',),
                                 max_iters=cfg.SOLVER.max_iters)
    models = build_model()
    optimizers, lr_decays = get_optimizers(models)

    mytrainer = SegnetTrainer(models, optimizers, lr_decays, dataset, sets=cfg.DATASETS.split, gpu_id=cfg.MODEL.gpu_id,
                              use_load_checkpoint=-1, net_type="LabelAE")

    net = mytrainer.train()


if __name__ == "__main__":
    main()
