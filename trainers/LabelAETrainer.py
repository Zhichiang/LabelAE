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
from utils.error_metrics import MAE
from utils.logger import setup_logger
from utils.label_embeding import one_hot_embed_2d
from utils.segmap_colorize import decode_segmap

from utils.loss import CrossEntropy2dLoss

from config import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name='co_depth_seg')

error_metrics = ['MAE']
val_error_metrics = ['pred_acc']
track_meters = ['loss_depth', 'val_loss']


class LabelAETrainer(BaseTrainer):
    def __init__(self, nets, optimizers, lr_schedulers, dataset, gpu_id,
                 sets=('train', 'val'), use_load_checkpoint=None, net_type="default", **kwargs):
        super(LabelAETrainer, self).__init__(nets, optimizers, lr_schedulers, objective=None,
                                             use_gpu=cfg.MODEL.use_gpu, workspace_dir=cfg.MODEL.chkpt_dir,
                                             net_type=net_type, gpu_id=gpu_id)

        self.encoder, self.decoder = nets
        self.optim_en, self.optim_de = optimizers

        self.sets = sets
        self.logger = setup_logger(type(self).__name__)
        self.logger.info("Trainer sets: " + str(self.sets))
        self.dataset = dataset
        self.use_load_checkpoint = use_load_checkpoint

        self.seg_loss = CrossEntropy2dLoss().to(self.device)

        self.val_criterion = dict(zip(val_error_metrics, [name for name in val_error_metrics]))
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

        for epoch in range(self.epoch, cfg.SOLVER.num_epochs + 1):
            self.epoch = epoch
            for lr_decay in self.lr_schedulers:
                lr_decay.step()

            # Add the average loss for this epoch to stats
            for s in self.sets:
                if s == 'train':
                    loss_meter = self.train_epoch()
                elif s == 'val':
                    loss_meter = self.validation()
                else:
                    loss_meter = None
                self.stats[s + '_loss'].append(loss_meter)

            # save checkpoint
            if self.use_save_checkpoint and self.epoch % cfg.SOLVER.save_chkpt_each == 0:
                self.save_checkpoint()
                self.logger.info('\nEpoch {}; ==> Checkpoint was saved successfully!'.format(self.epoch))

        if self.epoch == cfg.SOLVER.num_epochs + 1:
            self.logger.info('Training finished! Inferencing...')
            self.validation()
            self.write_visual_data()
        else:
            # save the final model
            if self.use_save_checkpoint:
                self.save_checkpoint()
                self.logger.info('\nEpoch {}; ==> Final Checkpoint was saved successfully!'.format(self.epoch))

        self.logger.info('Training finished!\n')
        return None

    def train_epoch(self):
        self.logger.info('train mode!')
        self.encoder.train(True)
        self.decoder.train(True)

        batch_num_per_epoch = len(self.dataset.loaders['train'])
        meters = dict(zip(track_meters, [AverageMeter() for _ in track_meters]))

        for batch_idx, data in enumerate(tqdm(self.dataset.loaders['train'])):
            data = self.data_to_device(data)
            input_rgb, input_seg = data
            input_seg = input_seg.long()

            # Encode the seg
            seg_onehot_tensor = one_hot_embed_2d(input_seg)

            self.optim_en.zero_grad()
            self.optim_de.zero_grad()
            seg_code = self.encoder(seg_onehot_tensor)
            rec_seg = self.decoder(seg_code)
            seg_loss = self.seg_loss(rec_seg, input_seg)                      # cross entropy loss
            seg_loss.backward()
            self.optim_en.step()
            self.optim_de.step()

            pred_map = F.softmax(rec_seg, dim=1).data.max(1)[1].cpu().numpy()
            color_pred = decode_segmap(pred_map)
            color_label = decode_segmap(input_seg.cpu().numpy())
            avg_seg_code = seg_code.sum(1)
            maxsum_seg_code = (seg_code > 0).int().sum(1).float() / cfg.MODEL.latent_len
            self.scalar_summary = {
                'train_loss/seg_loss': seg_loss.item(),
            }
            self.image_summary = {
                'train/input_rgb': input_rgb[0].clone().cpu().data,
                'train/color_pred': torch.tensor(color_pred.transpose((0, 3, 1, 2)))[0],
                'train/color_target': torch.tensor(color_label.transpose((0, 3, 1, 2)))[0],
                'train/avg_seg_code': avg_seg_code[0].clone().cpu().data,
                'train/maxsum_seg_code': maxsum_seg_code[0].clone().cpu().data,
            }

            # Save to summary writer
            self.writer.write_data(scalars=self.scalar_summary, images=self.image_summary,
                                   epoch=self.epoch, batch_idx=batch_idx,
                                   per_batch_s=cfg.SOLVER.writer_sample_each,
                                   per_batch_i=cfg.SOLVER.writer_sample_each * 100,
                                   is_print=False, is_train=True, batch_per_epoch=batch_num_per_epoch)
            meters['loss_depth'].update(seg_loss.item(), input_rgb.size(0))

        self.scalar_summary = {
            'train_loss_epoch/depth_loss_epoch': meters['loss_depth'].avg,
        }

        self.writer.write_data(scalars=self.scalar_summary, images=None,
                               epoch=self.epoch, batch_idx=-1, is_print=True, is_train=True)
        return deepcopy(meters)

    def validation(self):
        self.logger.info('validation mode!')
        self.encoder.eval()
        self.decoder.eval()

        batch_num_per_epoch = len(self.dataset.loaders['val'])
        metric_meter = dict(zip(val_error_metrics, [AverageMeter() for _ in range(len(val_error_metrics))]))
        meters = dict(zip(track_meters, [AverageMeter() for _ in track_meters]))

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.dataset.loaders['val'])):
                data = self.data_to_device(data)
                input_rgb, input_seg = data
                input_seg = input_seg.long()
                seg_onehot_tensor = one_hot_embed_2d(input_seg)

                seg_code = self.encoder(seg_onehot_tensor)
                rec_seg = self.decoder(seg_code)
                seg_loss = self.seg_loss(rec_seg, input_seg)  # cross entropy loss

                pred_map = F.softmax(rec_seg, dim=1).data.max(1)[1]
                diff_rate = float((torch.abs(pred_map - input_seg) > 0).sum().cpu()) / float(pred_map.numel())

                # Save to summary writer
                pred_map = pred_map.cpu().numpy()
                color_pred = decode_segmap(pred_map)
                color_label = decode_segmap(input_seg.cpu().numpy())
                color_diff = color_pred - color_label
                avg_seg_code = seg_code.sum(1)
                maxsum_seg_code = (seg_code > 0).int().sum(1).float() / cfg.MODEL.latent_len
                self.scalar_summary = {
                    'val_loss/seg_loss': seg_loss.item(),
                    'val_loss/diff_rate': diff_rate,
                }
                self.image_summary = {
                    'val/input_rgb': input_rgb[0].clone().cpu().data,
                    'val/color_pred': torch.tensor(color_pred.transpose((0, 3, 1, 2)))[0],
                    'val/color_target': torch.tensor(color_label.transpose((0, 3, 1, 2)))[0],
                    'val/color_diff': torch.tensor(color_diff.transpose((0, 3, 1, 2)))[0],
                    'val/avg_seg_code': avg_seg_code[0].clone().cpu().data,
                    'val/maxsum_seg_code': maxsum_seg_code[0].clone().cpu().data,
                }
                self.writer.write_data(scalars=self.scalar_summary, images=self.image_summary,
                                       epoch=self.epoch, batch_idx=batch_idx,
                                       per_batch_s=cfg.SOLVER.val_writer_sample_each,
                                       per_batch_i=cfg.SOLVER.val_writer_sample_each * 50,
                                       is_print=False, is_train=False, batch_per_epoch=batch_num_per_epoch)

                meters['val_loss'].update(seg_loss.item(), input_rgb.size(0))
                metric_meter['pred_acc'].update(diff_rate, input_rgb.size(0))

            self.scalar_summary = dict()
            for m in val_error_metrics:
                self.logger.info('[Val] Epoch {}; [{}]: {:.8f}'.format(self.epoch, m, metric_meter[m].avg))
                tag = 'val_epoch/' + m
                self.scalar_summary[tag] = metric_meter[m].avg
            self.scalar_summary['val_epoch/val_loss'] = meters['val_loss'].avg
            self.writer.write_data(scalars=self.scalar_summary, images=None,
                                   epoch=self.epoch, batch_idx=-1,
                                   is_print=True, is_train=False)

        return [deepcopy(meters), deepcopy(metric_meter)]

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

