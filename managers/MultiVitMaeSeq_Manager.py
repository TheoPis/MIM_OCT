import datetime
from tqdm import tqdm
from typing import Union, Dict
from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch import nn
from utils import to_numpy, printlog, save_qualitative_results, calculate_metrics
from .MultiVitMae_Manager import MultiVitMaeManager
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


class MultiVitMaeSeqManager(MultiVitMaeManager, ABC):

    @property
    def sequence_length(self):
        if hasattr(self.flat_model.backbone.encoder, 'patch_embeders'):
            seq_len = 0
            # some_modality = list(self.model.backbone.encoder.patch_embeders.keys())[0]
            # note: we assume all modalities have the same patch size
            for modality in self.flat_model.backbone.encoder.patch_embeders:
                seq_len += self.flat_model.backbone.encoder.patch_embeders[modality].num_patches
            return seq_len
        else:
            raise ValueError('Could not find patch_embeders MMAESeq type of model')

    def forward_train_step(self, data: Dict[str, torch.Tensor], reduce_batch: bool = True, **kwrargs):
        """ training involves random masking of one modality
        and reconstructing it by cross-attending to features from the other visible modality (and viceversa)
        """
        assert isinstance(data, dict), f'input data must be a dict, got {type(data)} with keys modality names'
        modalities = list(data.keys())
        # outs: dict with keys: "cls_token", "output", "mask"
        with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_autocast):
            ret = self.model(x=data, is_training=True)
            outputs = ret['output']   # ret={'output':Dict[modality,tensor], ...
            loss = 0
            for m in modalities:
                loss_m = self.get_loss(data[m], outputs[m], ret[f'mask_{m}'], reduce_batch=reduce_batch)
                ret[f'loss_m_{m}'] = loss_m
                loss += loss_m

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
        for batch_ind, batch in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
            # prepare batch
            data = batch
            modalities = batch.keys()
            for modality in modalities:
                data[modality] = data[modality].to(self.device, non_blocking=True)

            # forward and gradient accumulation
            grad_context = self.get_grad_context(accumulated_iter)

            losses_accum = {f'loss_m_{m}': torch.tensor(0.0, device=self.device, dtype=self.dtype) for m in modalities}
            losses_accum['loss'] = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            with grad_context:
                ret = self.forward_train_step(data)
                loss = ret['loss'] / self.grad_accumulation_steps

            losses_accum['loss'] += loss.detach()
            losses_accum.update({f'loss_m_{m}': losses_accum[f'loss_m_{m}'] + ret[f'loss_m_{m}'].detach() for m in modalities})

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_iter += 1
            if accumulated_iter % self.grad_accumulation_steps == 0:
                # reset accumulated_iter
                accumulated_iter = 0

                # average losses across devices
                if self.parallel:
                    for loss_term in losses_accum:
                        dist.all_reduce(losses_accum[loss_term], op=dist.ReduceOp.AVG)

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

                # time step
                torch.cuda.synchronize()
                dt = (datetime.datetime.now() - t0).total_seconds() * 1000
                t0 = datetime.datetime.now()
                tps = (self.batch_size * self.grad_accumulation_steps * self.sequence_length) / (dt/1000)

                # logging
                self.train_logging(batch_ind, data, losses_accum, tpb=dt, tps=tps, norm=norm, modality=modalities)

                # increment device-wise step
                self.global_step += 1

            if batch_ind == 2 and self.debugging:
                break

    def validate(self):
        self.model.eval()
        valid_loss = 0
        if self.rank == 0:
            printlog(f'\n starting validation for process rank {self.rank}')
        else:
            printlog(f'\n skipping validation for process rank {self.rank}')
            return

        if not self.parallel:
            torch.backends.cudnn.benchmark = False

        loss_dict = {m: 0 for m in self.config['data']['modality']}

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
                data = batch
                modalities = batch.keys()

                for modality in modalities:
                    data[modality] = data[modality].to(self.device, non_blocking=True)

                ret = self.forward_train_step(data, reduce_batch=False)
                loss = ret['loss']
                for m in modalities:
                    loss_dict[m] += ret[f'loss_m_{m}'].sum().item()
                valid_loss += loss.sum().item()

                if batch_index == 2 and self.debugging:
                    break
            # mean val loss
            num_images = len(self.data_loaders['valid_loader']) * self.config['data']['valid_batch_size']
            valid_loss /= num_images
            for m in modalities:
                loss_dict[m] /= num_images
            self.valid_logging(valid_loss, loss_dict=loss_dict)

    def train_logging(self, batch_num, data, losses: Dict[str, torch.Tensor], tpb, tps, norm, **kwargs):
        modalities_current = list(data.keys())
        # train logging
        if self.scheduler is not None and self.config['train']['lr_batchwise']:
            lr = self.scheduler.get_lr()[0]

        info_string = ""
        for modality in modalities_current:
            info_string += f"{modality} "

        if (self.rank == 0) and self.use_wandb and (self.global_step % self.config["logging"]['wandb_step'] == 0):
            wandb.log({"loss": losses['loss'].item(), "lr": lr, "t_per_batch": tpb, "norm": norm}, step=self.global_step)
            for m in modalities_current:
                wandb.log({f"loss_m_{m}": losses[f'loss_m_{m}'].item()})  # save modality-specific loss

        losses_string = f"Loss: {losses['loss'].item():.4f} "
        for m in modalities_current:
            losses_string += f"loss-{m}: {losses[f'loss_m_{m}'].item():.4f} "

        if self.global_step % self.config["logging"]['display_step'] == 0:
            current_epoch = self.epoch + self.start_epoch
            printlog(f"Epoch {current_epoch:03d} | step {self.global_step:06d} | Batch {batch_num:03d} "
                     f"| {losses_string} | grad_norm: {norm:.4f} "
                     f"| {info_string} dt: {tpb:.1f} | tokens/sec {tps:.1f} "
                     f"r {self.rank:01d}")

        if self.global_step % self.config["logging"]["train_img_log_step"] == 0:
            train_loader = self.data_loaders[self.train_schedule[self.epoch]]
            # generating qualitative results for both modalities
            # for loader in train_loader.dataloaders:
            #     modality = loader.dataset.modality
            #     if hasattr(loader.dataset, 'use_omnivision_api'):
            #         loader.dataset.use_omnivision_api = False  # turn off ominvision api

            temp = train_loader.dataset.return_metadata
            train_loader.dataset.return_metadata = False  # turn off metadata
            save_qualitative_results(loader=train_loader,
                                     num_images=self.config['logging']['max_valid_imgs'],
                                     manager=self,
                                     filename=f'visuals_train@{self.global_step}.png',
                                     tag='_multimae_seq')

            train_loader.dataset.return_metadata = temp   # restore return_metadata

    def valid_logging(self, valid_loss, **kwargs):
        """ logging - checkpoint saving - best val tracking
        Controlled by config using the following entries
             "logging": {
             "valid_freq": 1,
             "checkpoint_epoch": 25,
             "wandb": true,
             "wandb_step": 50,
             "display_step": 1,
             "train_img_log_step": 2500,
             "max_valid_imgs": 8
         }
         """
        loss_dict = kwargs.get('loss_dict', None)
        if valid_loss < self.best_loss:
            prev_best_loss = self.best_loss
            self.best_loss = valid_loss
            self.metrics.update({'best_val_loss': valid_loss,
                                 'best_loss_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            printlog("New best validation loss: {:.5f} prev {:.5f}".format(self.best_loss, prev_best_loss))
            self.save_checkpoint(save_as='best')  # fixme

        elif (self.epoch % self.config['logging']['checkpoint_epoch'] == 0) \
                and self.epoch > 0 or \
                ((self.epoch + self.start_epoch) == self.config['train']['epochs'] - 1):

            self.save_checkpoint()

        if self.epoch % self.config['logging']['val_image_log_epoch'] == 0:
            save_qualitative_results(self.data_loaders['valid_loader'],
                                     self.config['logging']['max_valid_imgs'],
                                     self,
                                     f'visuals_val_@{self.global_step}.png',
                                     tag='_multimae_seq')

        if (self.rank == 0) and self.use_wandb:
            wandb.log({"val_loss": valid_loss}, step=self.global_step-1)
            if loss_dict:
                for m in loss_dict:
                    self.metrics.update({f'val_loss_{m}': loss_dict[m]})
                    wandb.log({f"val_loss_{m}": loss_dict[m]}, step=self.global_step-1)

        self.write_info_json()

    def infer(self, show_with_validation_metrics=False, chkpt_type='best'):
        assert chkpt_type in ['final', 'best'], f"{chkpt_type} not in ['final', 'best']"
        self.model.eval()
        if self.use_ema:
            self.ema_model.average_model.eval()
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
            ret = self.forward_train_step(data, lbl, reduce_batch=False, skip_loss=True)
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
        printlog("=" * 25)
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

            printlog(f"{'=' * 10}_val_{'=' * 10} @ {chkpt_type}")
            printlog(metrics_val_show)
            printlog("=" * 25)
            latex_line = self.get_latex_line(metrics, metrics_val_show)
            printlog(latex_line)
            return metrics, metrics_val_show, latex_line
        else:
            return metrics, None, None
