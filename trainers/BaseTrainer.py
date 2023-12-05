import os
import glob
import torch
import torch.nn as nn

from trainers.Summary import Summary


class BaseTrainer(object):
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function"""

    def __init__(self, nets, optimizers, lr_schedulers, objective, gpu_id, use_gpu=True, workspace_dir=None,
                 net_type="default"):
        """net: model architecture
           workspace_dir: directory of workspace
        """

        self.nets = nets
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.objective = objective
        self.use_gpu = use_gpu
        self.workspace_dir = None
        self.use_save_checkpoint = workspace_dir is not None
        self.net_type = net_type
        self.gpu_id = gpu_id

        # prepare workspace
        if workspace_dir is not None:
            self.workspace_dir = os.path.expanduser(workspace_dir)
            if not os.path.exists(self.workspace_dir):
                os.makedirs(self.workspace_dir)

        self.writer = Summary()

        if self.use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            if len(gpu_id) == 1:
                for idx in range(len(self.nets)):
                    self.nets[idx] = self.nets[idx].to(torch.device('cuda'))
                self.device = next(self.nets[0].parameters()).device
            else:
                # device_ids = list(range(len(gpu_id)))
                device_ids = [0, 1]
                for idx in range(len(self.nets)):
                    self.nets[idx] = self.nets[idx].cuda(device_ids[0])
                    torch.manual_seed(59)
                    self.nets[idx] = nn.DataParallel(self.nets[idx], device_ids=device_ids)
                self.device = next(self.nets[0].parameters()).device

        self.epoch = 1
        self.stats = {}

    def train_epoch(self):
        raise NotImplementedError

    def train(self, max_epochs):
        """Do training for the given number of epochs."""

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            self.train_epoch()
            if self.use_save_checkpoint:
                self.save_checkpoint()

        print('Finished training!')

    def save_checkpoint(self):
        """Save a checkpoint of the network, optimizer, lr_scheduler and other varaibles."""
        nets_state_dict = []
        optim_state_dict = []
        lr_decay_state_dict = []
        for net in self.nets:
            nets_state_dict.append(net.state_dict())
        for optim in self.optimizers:
            optim_state_dict.append(optim.state_dict())
        for lr_decay in self.lr_schedulers:
            lr_decay_state_dict.append(lr_decay.state_dict())

        state = {
            'epoch': self.epoch,
            'net_type': self.net_type,
            'net': nets_state_dict,
            'optimizer': optim_state_dict,
            'lr_scheduler': lr_decay_state_dict,
            'stats': self.stats,
            'use_gpu': self.use_gpu,
        }

        chkpt_path = os.path.join(self.workspace_dir, 'checkpoints')
        if not os.path.exists(chkpt_path):
            os.makedirs(chkpt_path)

        file_path = '{}/{}_epoch{:04d}.pth.tar'.format(chkpt_path, self.net_type, self.epoch)
        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None):
        """Load a checkpoint file.
        Can be called by three different ways:
            load_checkpoint(): Loads the latest checkpoint from the worksapce. Use this to continue training.
            load_checkpoint(epoch_num): Loads the network at the given epoch num (int).
            load_checkpoint(path_to_checkpoint): Loads the network from the absolute path (str).
        """

        chkpt_path = os.path.join(self.workspace_dir, 'checkpoints')

        if checkpoint is None:
            checkpoint_list = sorted(glob.glob('{}/{}_epoch*.pth.tar'.format(chkpt_path, self.net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found!\n')
                return False
        elif isinstance(checkpoint, int):
            checkpoint_path = '{}/{}_epoch{:04d}.pth.tar'.format(chkpt_path, self.net_type, checkpoint)
        elif isinstance(checkpoint, str):
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        if self.use_gpu:
            checkpoint_dict = torch.load(checkpoint_path)
        else:
            checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert self.net_type == checkpoint_dict['net_type'], 'Wrong Network Type!'
        self.epoch = checkpoint_dict['epoch'] + 1
        for idx in range(len(self.nets)):
            self.nets[idx].load_state_dict(checkpoint_dict['net'][idx])
        for idx in range(len(self.optimizers)):
            self.optimizers[idx].load_state_dict(checkpoint_dict['optimizer'][idx])
        if 'lr_scheduler' in checkpoint_dict:
            for idx in range(len(self.lr_schedulers)):
                self.lr_schedulers[idx].load_state_dict(checkpoint_dict['lr_scheduler'][idx])
                self.lr_schedulers[idx].last_epoch = checkpoint_dict['epoch']
        self.stats = checkpoint_dict['stats']

        return True

    def data_to_device(self, data: tuple):
        ret = list()
        for d in data:
            if d is None:
                ret.append(None)
                continue
            ret.append(d.to(self.device))
        return ret
