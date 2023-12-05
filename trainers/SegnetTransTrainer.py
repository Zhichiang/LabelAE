#####################################
__author__ = "Zhichiang"
#####################################

from copy import deepcopy
import torch

import pprint
import logging

from tqdm import tqdm
import torch.nn.functional as F

from trainers.BaseTrainer import BaseTrainer
from utils.AverageMeter import AverageMeter
from utils.error_metrics import MAE, IoURunningScore
from utils.logger import setup_logger
from utils.segmap_colorize import decode_segmap

from utils.loss import CriterionDSN, GANLoss

from modules.LabelAE.LabelAEv2 import LabelDecoder

from config import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='co_depth_seg')

error_metrics = ['MAE']
val_error_metrics = ['MAE', 'IoURunningScore']
track_meters = ['val_loss']


class SegnetTrainer(BaseTrainer):
    def __init__(self, nets, optimizers, lr_schedulers, dataset, gpu_id,
                 sets=('train', 'val'), use_load_checkpoint=None, net_type="default", **kwargs):
        super(SegnetTrainer, self).__init__(nets, optimizers, lr_schedulers, objective=None,
                                            use_gpu=cfg.MODEL.use_gpu, workspace_dir=cfg.MODEL.chkpt_dir,
                                            net_type=net_type, gpu_id=gpu_id)

        self.segnet = nets[0]
        self.optim_seg = optimizers[0]
        self.lr_decay_seg = lr_schedulers[0]

        self.dataset = dataset
        self.dataset_train = iter(dataset.loaders['train'])

        self.sets = sets
        self.logger = setup_logger(type(self).__name__)
        self.logger.info("Trainer sets: " + str(self.sets))
        self.use_load_checkpoint = use_load_checkpoint

        self.seg_loss = CriterionDSN().to(self.device)
        self.val_metrics = IoURunningScore(19)
        self.upsample_512 = torch.nn.Upsample(size=[512, 1024], mode='bilinear')

        self.scalar_summary = dict()
        self.image_summary = dict()

        for s in self.sets:
            self.stats[s + '_loss'] = []
        for metric in error_metrics:
            self.stats[metric] = []

        # load latest checkpoint
        if self.use_load_checkpoint is not None:
            if self.use_load_checkpoint > 0:
                self.logger.info('\n=> Loading checkpoint{} ...'.format(self.use_load_checkpoint))
                if self.load_checkpoint(self.use_load_checkpoint):
                    self.logger.info('Epoch {}; Checkpoint {} was loaded successfully!\n'.format(
                        self.epoch, self.use_load_checkpoint))
            elif self.use_load_checkpoint == -1:
                self.logger.info('\n=> Loading latest checkpoint ...')
                if self.load_checkpoint():
                    self.logger.info('Latest checkpoint {} was loaded successfully!\n'.format(self.use_load_checkpoint))

    def train(self, *args):  # 18621887380
        logger.info(pprint.pformat(cfg))

        for epoch in tqdm(range(self.epoch, cfg.SOLVER.max_iters + 1), ascii=True):
            self.epoch = epoch
            self.segnet.train(True)
            self.lr_decay_seg.step()

            try:
                data = next(self.dataset_train)
            except StopIteration:
                print("in iteration {}, catch stop_iteration".format(epoch))
                self.dataset_train = iter(self.dataset.loaders['train'])
                data = next(self.dataset_train)
            data = self.data_to_device(data)
            input_rgb, input_seg = data
            input_seg = input_seg.long()

            rec_seg = self.segnet(input_rgb)
            seg_loss = total_loss = self.seg_loss(rec_seg, input_seg)  # cross entropy loss

            self.optim_seg.zero_grad()
            total_loss.backward()
            self.optim_seg.step()

            # ############################## Visualization ############################## #
            pred_map = F.softmax(rec_seg[0], dim=1).data.max(1)[1].cpu().numpy()
            color_pred = decode_segmap(pred_map)
            color_label = decode_segmap(input_seg.cpu().numpy())
            self.scalar_summary = {
                'train_loss/seg_loss': seg_loss.item(),
            }
            self.image_summary = {
                'train/input_rgb': input_rgb[0].clone().cpu().data,
                'train/color_pred': torch.tensor(color_pred.transpose((0, 3, 1, 2)))[0],
                'train/color_target': torch.tensor(color_label.transpose((0, 3, 1, 2)))[0],
            }

            # Save to summary writer
            self.writer.write_data(scalars=self.scalar_summary, images=self.image_summary,
                                   epoch=1, batch_idx=epoch, per_batch_s=cfg.SOLVER.writer_sample_each,
                                   per_batch_i=cfg.SOLVER.writer_sample_each*100,
                                   is_print=False, is_train=True, batch_per_epoch=1)

            if epoch % cfg.SOLVER.val_calc_each == 0:
                self.logger.info('validation mode!')
                self.segnet.eval()

                meters = dict(zip(track_meters, [AverageMeter() for _ in track_meters]))

                with torch.no_grad():
                    for batch_idx, data in enumerate(self.dataset.loaders['val']):
                        batch_num_per_epoch = len(self.dataset.loaders['val'])
                        if int(batch_idx / batch_num_per_epoch * 100) % 10 == 0:
                            print("{:02d}%".format(int(batch_idx / batch_num_per_epoch * 100)),
                                  end=" ", flush=True)
                        data = self.data_to_device(data)
                        input_rgb, input_seg = data
                        input_seg = input_seg.long()

                        seg_pred = self.segnet(input_rgb)
                        seg_loss = self.seg_loss(seg_pred, input_seg)  # cross entropy loss

                        seg_pred = self.upsample_512(seg_pred[0])
                        _pred = seg_pred.data.max(1)[1].cpu().numpy()
                        gt = input_seg.data.cpu().numpy()
                        self.val_metrics.update(gt, _pred)

                        # Save to summary writer
                        pred_map = F.softmax(seg_pred, dim=1).data.max(1)[1].cpu().numpy()
                        color_pred = decode_segmap(pred_map)
                        color_label = decode_segmap(input_seg.cpu().numpy())
                        color_diff = color_pred - color_label
                        self.scalar_summary = {
                            'val_loss/seg_loss': seg_loss.item(),
                        }
                        self.image_summary = {
                            'val/input_rgb': input_rgb[0].clone().cpu().data,
                            'val/color_pred': torch.tensor(color_pred.transpose((0, 3, 1, 2)))[0],
                            'val/color_target': torch.tensor(color_label.transpose((0, 3, 1, 2)))[0],
                            'val/color_diff': torch.tensor(color_diff.transpose((0, 3, 1, 2)))[0],
                        }
                        self.writer.write_data(scalars=self.scalar_summary, images=self.image_summary,
                                               epoch=int(epoch / cfg.SOLVER.val_calc_each), batch_idx=batch_idx,
                                               per_batch_s=cfg.SOLVER.val_writer_sample_each,
                                               per_epoch=1, per_batch_i=cfg.SOLVER.val_writer_sample_each*50,
                                               is_print=False, is_train=False, batch_per_epoch=batch_num_per_epoch)

                        meters['val_loss'].update(seg_loss.item(), input_rgb.size(0))

                    cty_score, cty_class_iou = self.val_metrics.get_scores()
                    self.scalar_summary = dict()
                    for m in track_meters:
                        self.logger.info('[Val] Epoch {}; [{}]: {:.8f}'.format(self.epoch, m, meters[m].avg))
                        tag = 'val_epoch/' + m
                        self.scalar_summary[tag] = meters[m].avg
                    self.scalar_summary['val_epoch/IoU'] = cty_score['Mean IoU : \t']
                    self.writer.write_data(scalars=self.scalar_summary, images=None,
                                           epoch=int(epoch / cfg.SOLVER.val_calc_each), batch_idx=-1,
                                           is_print=True, is_train=False)
                    self.val_metrics.reset()

            # save checkpoint
            if self.use_save_checkpoint and self.epoch % cfg.SOLVER.save_chkpt_each == 0:
                self.save_checkpoint()
                self.logger.info('\nEpoch {}; ==> Checkpoint was saved successfully!'.format(self.epoch))

        if self.epoch == cfg.SOLVER.num_epochs + 1:
            self.logger.info('Training finished! Inferencing...')
            # self.validation()
            self.write_visual_data()
        else:
            # save the final model
            if self.use_save_checkpoint:
                self.save_checkpoint()
                self.logger.info('\nEpoch {}; ==> Final Checkpoint was saved successfully!'.format(self.epoch))

        self.logger.info('Training finished!\n')
        return None

    def write_visual_data(self):
        train_losses = self.stats['train_loss']
        val_losses = self.stats['val_loss']
        for idx, epoch_data in enumerate(train_losses):
            self.scalar_summary = dict()
            for m in epoch_data:
                if 'val' in m:
                    continue
                tag = '-train_epoch/' + m
                self.scalar_summary[tag] = epoch_data[m].avg

            self.writer.write_data(scalars=self.scalar_summary, images=None,
                                   epoch=idx, batch_idx=-1)

        for idx, epoch_data in enumerate(val_losses):
            meters, metric_meter = epoch_data
            self.scalar_summary = dict()
            for m in meters:
                if 'val' in m:
                    tag = '-val_epoch/' + m
                    self.scalar_summary[tag] = meters[m].avg
            for m in metric_meter:
                if 'val' in m:
                    tag = '-val_epoch/' + m
                    self.scalar_summary[tag] = metric_meter[m].avg

            self.writer.write_data(scalars=self.scalar_summary, images=None,
                                   epoch=idx, batch_idx=-1)

        pass

