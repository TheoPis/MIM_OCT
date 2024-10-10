import datetime
import wandb
from abc import ABC
from typing import Union
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist

import torch
from torch import nn

from managers.BaseManager import BaseManager
from utils import to_numpy, printlog, save_qualitative_results, calculate_metrics

ParallelType = nn.parallel.DistributedDataParallel
Model = Union[nn.Module, ParallelType]


class VitManager(BaseManager, ABC):
    @property
    def patch_shape(self):
        return self.flat_model.backbone.trunk.patch_embed.patch_size

    @property
    def sequence_length(self):
        if hasattr(self.flat_model.backbone.encoder, 'patch_embeders'):
            some_modality = list(self.flat_model.backbone.encoder.patch_embeders.keys())[0]
            return self.flat_model.backbone.encoder.patch_embeders[some_modality].num_patches
        elif hasattr(self.flat_model.backbone.encoder, 'patch_embed'):
            return self.flat_model.backbone.encoder.patch_embed.num_patches
        else:
            raise ValueError(f"Neither patch_embed not patch_embeders attribute found for model of type {type(self.model)}")

    def forward_train_step(self, img, lbl, reduce_batch=True, **kwrargs):
        """
        :param img: [B, C, H, W] tensor
        :param lbl: [B, num_classes] tensor
        :param reduce_batch: whether to reduce the batch dimension in the loss
        :return:
        """
        ret = dict()
        with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_autocast):
            output = self.model(img)
            loss = self.get_loss(img, output, lbl, reduce_batch=reduce_batch)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def forward_val_step(self, img, lbl, skip_loss=False, **kwrargs):
        """
        :param img: [B, C, H, W] tensor
        :param lbl: [B, num_classes] tensor
        :param skip_loss: whether to skip the loss calculation
        :return:
        """
        ret = dict()
        if self.use_ema and self.ema_model is not None:
            output = self.ema_model(img)
        else:
            output = self.model(img)

        loss = None
        if not skip_loss:
            loss = self.get_loss(img, output, lbl, reduce_batch=False)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def get_loss(self, img, output, lbl, reduce_batch=True):
        """ apply a supervised loss using img and labels (hence img and reduce_batch are ignored)
        for Vit_Manager or chidlren that do not override this, it is expected that
        the task is detection or classification
        img: [B, C, H, W]
        output: [B, N_classes] (N_classes according to classification/detection task)
        lbl: [B, N_classes] (N_classes according to classification/detection task)
        reduce_batch: whether to reduce the batch dimension in the loss
        """
        loss = self.loss(output, lbl)
        return loss

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        self.optimiser.zero_grad()
        t0 = datetime.datetime.now()
        accumulated_iter = 0
        for batch_ind, batch in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
            batch_dict = self.unpack_batch(batch)
            img, lbl, metadata = batch_dict['img'], batch_dict['lbl'], batch_dict['metadata']
            img = img.to(self.device, non_blocking=True)
            lbl = lbl.to(self.device, non_blocking=True)

            # forward and gradient accumulation
            grad_context = self.get_grad_context(accumulated_iter)
            loss_accum = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            with grad_context:
                ret = self.forward_train_step(img, lbl)
                loss = ret['loss'] / self.grad_accumulation_steps  # denominator is accumulation_steps * batch_size
                output = ret['output']
            loss_accum += loss.detach()
            #
            if self.debugging:printlog(f"step= {self.global_step} accumulated_iter "
                                       f"{accumulated_iter} | grad_accumulation_steps {self.grad_accumulation_steps}")

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_iter += 1
            if accumulated_iter % self.grad_accumulation_steps == 0:
                # reset accumulated_iter
                accumulated_iter = 0

                # average loss across devices
                if self.parallel:
                    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

                # gradient clipping
                norm = 0
                if self.grad_norm_clip is not None:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimiser)  # unscale before doing clipping
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)

                # update model
                if self.scaler is not None:
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    self.optimiser.step()

                self.optimiser.zero_grad()
                # lr scheduler step
                if self.scheduler is not None and self.config['train']['lr_batchwise']:
                    self.scheduler.step()
                # ema step
                if (self.rank == 0) and self.use_ema and self.ema_model is not None:
                    self.ema_model.update()

                # time step
                torch.cuda.synchronize()
                dt = (datetime.datetime.now() - t0).total_seconds() * 1000
                t0 = datetime.datetime.now()
                tps = (self.batch_size * self.grad_accumulation_steps * self.sequence_length) / (dt/1000)

                # logging
                self.train_logging(batch_ind, output, img, loss_accum.item(), dt, tps, norm)

                # increment device-wise step
                self.global_step += 1

                if batch_ind == 2 and self.debugging:
                    break

    @torch.no_grad()
    def validate(self):
        if not (self.rank == 0):
            return

        self.model.eval()
        if self.use_ema:
            self.ema_model.average_model.eval()

        valid_loss = 0
        if not self.parallel:
            torch.backends.cudnn.benchmark = False

        predictions = []
        labels = []
        self.data_loaders['valid_loader'].dataset.image_set = None
        for batch_index, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
            batch_dict = self.unpack_batch(batch)
            img, lbl, metadata = batch_dict['img'], batch_dict['lbl'], batch_dict['metadata']
            img = img.to(self.device, non_blocking=True)
            lbl = lbl.to(self.device, non_blocking=True)

            # forward
            ret = self.forward_val_step(img, lbl, reduce_batch=False)
            loss = ret['loss']
            pred = ret['output']

            # collect results and validation loss
            predictions.append(to_numpy(pred))
            labels.append(to_numpy(lbl))
            valid_loss += loss.sum().item()
            if batch_index == 10 and self.debugging:
                break
        valid_loss /= len(self.data_loaders['valid_loader'].dataset)
        metrics = calculate_metrics(labels, predictions, dataset=self.dataset)  # this is a numpy function
        self.valid_logging(valid_loss, metrics)

    def train_logging(self, batch_num, output, img, loss, dt, tps, norm):
        """ Log training info to console and wandb
            to extend to other metric add wandb_dict.update({"name_of_metric": metric_scalar_value})
        :arg batch_num: integer batch index
        :arg output
        :arg img
        :arg loss: loss tensor
        :arg dt is time per batch in ms
        :arg tps is throughput per second
        :arg norm is the gradient norm
        """
        info_string = ""  # extra info that will be printed to console
        wandb_dict = {}  # dictionary to be logged to wandb
        lr = self.scheduler.get_last_lr()[-1] if self.scheduler is not None else self.config['train']['learning_rate']
        # basic info
        if self.use_wandb and (self.global_step % self.config["logging"]["wandb_step"] == 0) and (self.rank == 0):
            wandb_dict.update({"loss": loss, "lr": lr, "t_per_batch": dt, 'norm': norm})
            wandb.log(wandb_dict, step=self.global_step)
        if self.global_step % self.config["logging"]['display_step'] == 0:
            current_epoch = self.epoch + self.start_epoch
            printlog(f"Epoch {current_epoch:03d} | step {self.global_step:06d} | Batch {batch_num:03d} "
                     f"| Loss: {loss:.4f} | grad_norm: {norm:.4f} | {info_string} dt: {dt:.1f} | tokens/sec {tps:.1f} "
                     f"r {self.rank:01d}")

    def valid_logging(self, valid_loss: Union[float, None], latest_metrics: dict, chkpt_type='last', visual_tag=None):
        """ logging val loss - checkpoint saving - best val tracking
        if config.mode = 'inference' then only l1ogs metrics and save images
        :arg valid_loss:
        :arg latest_metrics: dict of metrics computed during validation
        :arg chkpt_type: 'last' or 'best'
        :arg visual_tag: tag for visualization function to save qualitative results
        :return:
        """
        if self.config['mode'] == 'training':
            self.metrics['final_val_loss'] = valid_loss
            self.metrics['final_epoch_step'] = [self.epoch + self.start_epoch, self.global_step - 1]

            for name, value in latest_metrics.items():
                self.metrics[f'final_{name}'] = value
                self.metrics[f'final_{name}_epoch_step'] = [self.epoch + self.start_epoch, self.global_step - 1]

            metric_current = latest_metrics[self.model_selection_metric]
            metric_best = self.metrics[f'best_{self.model_selection_metric}']
            if metric_best < metric_current:  # we assume higher metric is better
                metric_best_prev = metric_best  # save previous best for printing
                # update metrics dict
                self.metrics.update({f'best_{self.model_selection_metric}': metric_current,
                                     f'best_{self.model_selection_metric}_epoch_step':
                                         [self.epoch + self.start_epoch, self.global_step - 1]})
                printlog("New best validation {}: {:.5f} prev {:.5f}".format(self.model_selection_metric,
                                                                             metric_current,
                                                                             metric_best_prev))
                # record metrics @best_self.model_selection_metric
                for name, value in latest_metrics.items():
                    if name not in [self.model_selection_metric]:
                        # keep at most 2 floating points from value
                        value = round(value, 4)
                        self.metrics[f'best_{name}'] = value
                printlog("***"*10)
                printlog(self.metrics)

                self.save_checkpoint(save_as='best')

            elif self.epoch % self.config['log_every_n_epochs'] == 0 and self.epoch > 0 or \
                    (self.epoch + self.start_epoch) == self.config['train']['epochs'] - 1:
                self.save_checkpoint()

            if self.use_wandb:
                # dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'MAP': MAP, 'mF1': mf1, 'MF1': Mf1}
                for k, v in latest_metrics.items():
                    wandb.log({k: v}, step=self.global_step-1)
                wandb.log({"val_loss": valid_loss}, step=self.global_step-1)

        elif self.config['mode'] == 'inference':
            if (self. rank == 0) and (self.config['mode'] == 'inference'):
                if self.use_wandb:
                    # dict_metrics = {'ROC AUC': roc_auc, 'mAP': mAP, 'MAP': MAP, 'mF1': mf1, 'MF1': Mf1}
                    for k, v in latest_metrics.items():
                        if chkpt_type == 'last':
                            wandb.log({k: v}, step=self.global_step - 1)
                        else:
                            wandb.log({k: v}, step=self.global_step + 1)

                latest_metrics.update({'chkpt_name': self.checkpoint_name})
                latest_metrics.update({"using_ema": self.use_ema})
                ema = '_ema' if self.use_ema else ''
                self.write_dict_json(config=latest_metrics,
                                     filename=
                                     f'{self.date}_eval_{self.config["data"]["split"][-1]}_{self.checkpoint_name}{ema}')
        self.write_info_json()

    @torch.no_grad()
    def infer(self, show_with_validation_metrics=False, chkpt_type='final'):
        assert chkpt_type in ['final', 'best'], f"{chkpt_type} not in ['final', 'best']"
        self.model.eval()
        if self.use_ema:
            self.ema_model.average_model.eval()

        if not self.parallel:
            torch.backends.cudnn.benchmark = False

        predictions = []
        labels = []
        self.data_loaders['valid_loader'].dataset.image_set = None
        self.data_loaders['valid_loader'].dataset.label_set = None
        for batch_index, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):

            if len(batch) == 2:
                img, lbl = batch
            else:
                img, lbl, metadata = batch

            img = img.to(self.device, non_blocking=True)
            lbl = lbl.to(self.device, non_blocking=True)

            # forward
            ret = self.forward_val_step(img, lbl, reduce_batch=False, skip_loss=True)
            pred = ret['output']

            # save results
            predictions.append(to_numpy(pred))
            labels.append(to_numpy(lbl))

            if batch_index == 4 and self.debugging:
                break

        # mean val loss
        metrics = calculate_metrics(labels, predictions, dataset=self.dataset)  # this is a numpy function
        self.valid_logging(None, metrics)

        printlog(f"{'=' * 10}_{self.config['data']['split'][-1]}_{'=' * 10} @ {chkpt_type}")
        printlog(metrics)  # test metrics
        printlog("="*25)
        chkpt_type += '_'
        if show_with_validation_metrics:
            metrics_val_show = {}
            # combine metrics (obtained on test split) with metrics_val (obtained on val split)
            metrics_val = self.metrics.copy()
            # printlog(f'{list(metrics_val.keys())}')
            for metric in metrics_val:
                m = metric.split(chkpt_type)[-1]
                # printlog(m)
                if m in metrics or metric in ['epoch', 'global_step']:
                    metrics_val_show[metric.split(chkpt_type)[-1]] = metrics_val[metric]

            printlog(f"{'='*10}_val_{'='*10} @ {chkpt_type}")
            printlog(metrics_val_show)
            printlog("="*25)
            latex_line = self.get_latex_line(metrics, metrics_val_show)
            printlog(latex_line)
            return metrics, metrics_val_show, latex_line
        else:
            return metrics, None, None

