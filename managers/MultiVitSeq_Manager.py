import datetime
from tqdm import tqdm
from contextlib import nullcontext
from typing import Union, Dict, List
from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch import nn
from utils import to_numpy, printlog, save_qualitative_results, calculate_metrics
from .VitMae_Manager import VitMaeManager
from .Vit_Manager import VitManager
import torch.distributed as dist
plt.gray()

ParallelType = nn.parallel.DistributedDataParallel
Model = Union[nn.Module, ParallelType]


def tensor2pil_show_label_color(x):
    # x: (B, C, H, W), plots 1st element of the batch
    from PIL import Image
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)
    assert len(x.shape) == 4
    x = to_numpy(torch.clip(x*255, 0, 255)).astype(np.uint8)
    if x.shape[1] == 1:
        x = x[0, 0, ...]
    else:
        x = x[0]
        x = np.transpose(x, (1, 2, 0))
    # only show 1st image in the batch
    Image.fromarray(x).show()


class MultiVitSeqManager(VitManager, ABC):
    """ Manager for MultiVitSeq models for multimodal classification tasks (e.g. DR proliferation classification) """

    @property
    def modalities(self):
        return self.config['data']['modality']

    @property
    def modes(self) -> Dict[str, List[str]]:
        return dict(zip(self.modalities, self.config['data']['mode']))

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

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        self.optimiser.zero_grad()
        accumulated_iter = 0  # used if gradient accumulation is used else has no effect
        t0 = datetime.datetime.now()
        for batch_ind, batch in enumerate(self.data_loaders['train_loader']):
            if isinstance(batch, (list, tuple)):
                batch, metadata = batch[0], batch[1]
            lbl = batch['label'].to(self.device, non_blocking=True)
            data = {}  # move to gpu
            # todo decide how to handle modes for each modality
            for modality in self.modalities:
                mode = self.modes[modality][0]  # only use the first mode for now fixme
                data.update({modality: batch[modality][mode].to(self.device, non_blocking=True)})

            # # forward
            # ret = self.forward_train_step(data, lbl)
            # loss = ret['loss']
            # output = ret['output']

            # forward and gradient accumulation
            grad_context = self.get_grad_context(accumulated_iter)
            loss_accum = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            with grad_context:
                ret = self.forward_train_step(data, lbl)
                loss = ret['loss'] / self.grad_accumulation_steps  # denominator is accumulation_steps * batch_size
                output = ret['output']
            loss_accum += loss.detach()


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
                self.train_logging(batch_ind, output, data, loss_accum.item(), dt, tps, norm)

                # increment device-wise step
                self.global_step += 1

                if self.debugging and batch_ind == 10:
                    break
            #     self.train_logging(batch_ind, output, data, loss, dt)
            #
            #     # increment device-wise step
            #     self.global_step += 1
            #
            # if self.debugging and batch_ind == 2:
            #     break

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        if self.use_ema:
            printlog("!! setting ema model to eval mode for validation !!")
            self.ema_model.average_model.eval()
        valid_loss = 0
        if self.rank == 0:
            printlog(f'\n starting validation for process rank {self.rank}')
        else:
            printlog(f'\n skipping validation for process rank {self.rank}')
            return

        if not self.parallel:
            torch.backends.cudnn.benchmark = False

        predictions = []
        labels = []

        for batch_index, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
            if isinstance(batch, (list, tuple)):
                batch, metadata = batch[0], batch[1]
            lbl = batch['label'].to(self.device, non_blocking=True)
            data = {}  # move to gpu
            for modality in self.modalities:
                mode = self.modes[modality][0]  # only use the first mode for now fixme
                data.update({modality: batch[modality][mode].to(self.device, non_blocking=True)})
            # forward
            ret = self.forward_train_step(data, lbl, reduce_batch=False)
            loss = ret['loss']
            pred = ret['output']
            # collect predictions and labels for metrics
            predictions.append(to_numpy(pred))
            labels.append(to_numpy(lbl))
            valid_loss += loss.sum().item()
            if batch_index == 30 and self.debugging:
                break

        valid_loss /= len(self.data_loaders['valid_loader'].dataset)
        metrics = calculate_metrics(labels, predictions, dataset=self.dataset)  # this is a numpy function
        self.valid_logging(valid_loss, metrics, visual_tag='_multiseq_prolif' if self.dataset == 'DR' else None)

    def train_logging(self, batch_num, output, img, loss, dt, tps, norm):
        """Logging during training
        :param batch_num: batch number
        :param output: model output
        :param img: input data
        :param loss: loss
        :param dt: time taken for the batch
        :param tps: tokens per second
        :param norm: gradient norm
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

    @torch.no_grad()
    def infer(self, show_with_validation_metrics=False, chkpt_type='best'):
        assert chkpt_type in ['final', 'best'], f"{chkpt_type} not in ['final', 'best']"
        self.model.eval()
        # if self.use_ema:
        #     printlog("!! setting ema model to eval mode for inference !!")
        #     self.ema_model.average_model.eval()
        if not self.parallel:
            torch.backends.cudnn.benchmark = False

        predictions = []
        labels = []
        ind = 0
        for batch_index, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
            lbl = batch['label'].to(self.device, non_blocking=True)
            data = {}  # move to gpu
            for modality in self.modalities:
                mode = self.modes[modality][0]  # only use the first mode for now fixme
                data.update({modality: batch[modality][mode].to(self.device, non_blocking=True)})
            # forward
            ret = self.forward_val_step(data, lbl, reduce_batch=False, skip_loss=True)
            pred = ret['output']
            # save results
            predictions.append(to_numpy(pred))
            labels.append(to_numpy(lbl))
            if batch_index == 250 and self.debugging:
                break

        # mean val loss
        metrics = calculate_metrics(labels, predictions, dataset=self.dataset)  # this is a numpy function
        self.valid_logging(None, metrics)

        printlog(f"{'=' * 10}_{self.config['data']['split'][-1]}_{'=' * 10}")
        printlog(metrics)
        printlog("="*25)
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