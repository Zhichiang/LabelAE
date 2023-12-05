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

from utils.loss import CrossEntropy2dLoss, CriterionDSN, GANLoss

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

        self.segnet, self.decoder, self.d_clf = nets
        self.optim_seg, self.optim_decoder, self.optim_dclf = optimizers
        self.lr_decay_seg, self.lr_decay_decoder, self.lr_decay_dclf = lr_schedulers

        if cfg.MODEL.pretrained:
            checkpoint_dict = torch.load("./pretrained/labelaev2_epoch75.pth.tar")
            self.decoder.load_state_dict(checkpoint_dict['net'][1])
            for p in self.decoder.parameters():
                p.requires_grad = False

        self.decoder = self.decoder.to(self.device)

        self.dataset = dataset
        self.dataset_train = iter(dataset.loaders['train'])
        self.dataset_train_t = iter(dataset.loaders['train_t'])

        self.sets = sets
        self.logger = setup_logger(type(self).__name__)
        self.logger.info("Trainer sets: " + str(self.sets))
        self.use_load_checkpoint = use_load_checkpoint

        self.seg_loss = CrossEntropy2dLoss().to(self.device)
        self.gan_loss = GANLoss().to(self.device)
        self.val_metrics = IoURunningScore(19)

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
            self.lr_decay_dclf.step()
            self.lr_decay_seg.step()
            if not cfg.MODEL.pretrained:
                self.lr_decay_decoder.step()

            try:
                data = next(self.dataset_train)
            except StopIteration:
                print("in iteration {}, catch stop_iteration".format(epoch))
                self.dataset_train = iter(self.dataset.loaders['train'])
                data = next(self.dataset_train)
            data = self.data_to_device(data)
            input_rgb, input_seg = data
            input_seg = input_seg.long()

            try:
                data_t = next(self.dataset_train_t)
            except StopIteration:
                print("in iteration {}, catch stop_iteration".format(epoch))
                self.dataset_train_t = iter(self.dataset.loaders['train_t'])
                data_t = next(self.dataset_train_t)
            data_t = self.data_to_device(data_t)
            input_rgb_t, input_seg_t = data_t

            seg_code = self.segnet(input_rgb)
            seg_code_t = self.segnet(input_rgb_t)

            # ############################## Discriminator ############################## #
            if cfg.MODEL.domainada:
                for p in self.d_clf.parameters():
                    p.requires_grad = True
                real_loss = self.gan_loss(self.d_clf(seg_code.detach()), real=True, loss_type='vgan-bce')
                fake_loss = self.gan_loss(self.d_clf(seg_code_t.detach()), real=False, loss_type='vgan-bce')
                seg_dis_loss = real_loss + fake_loss
                self.optim_dclf.zero_grad()
                seg_dis_loss.backward()
                self.optim_dclf.step()
                for p in self.d_clf.parameters():
                    p.requires_grad = False
            else:
                seg_dis_loss = torch.zeros(1).to(self.device)

            # ############################## Generator ############################## #
            if cfg.MODEL.domainada:
                sim_loss = self.gan_loss(self.d_clf(seg_code_t), real=True, loss_type='vgan-bce')
            else:
                sim_loss = torch.zeros(1).to(self.device)

            rec_seg = self.decoder(seg_code)
            seg_loss = self.seg_loss(rec_seg, input_seg)  # cross entropy loss
            total_loss = sim_loss * cfg.SOLVER.lambda_fd + seg_loss

            if not cfg.MODEL.pretrained:
                self.optim_decoder.zero_grad()
            self.optim_seg.zero_grad()
            total_loss.backward()
            self.optim_seg.step()
            if not cfg.MODEL.pretrained:
                self.optim_decoder.step()

            # ############################## Visualization ############################## #
            pred_map = F.softmax(rec_seg, dim=1).data.max(1)[1].cpu().numpy()
            color_pred = decode_segmap(pred_map)
            color_label = decode_segmap(input_seg.cpu().numpy())
            self.scalar_summary = {
                'train_loss/seg_loss': seg_loss.item(),
                'train_loss/seg_dis_loss': seg_dis_loss.item(),
                'train_loss/sim_loss': sim_loss.item(),
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

                        seg_code = self.segnet(input_rgb)
                        seg_pred = self.decoder(seg_code)
                        seg_loss = self.seg_loss(seg_pred, input_seg)  # cross entropy loss

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

