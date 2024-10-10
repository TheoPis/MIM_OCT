import datetime
from tqdm import tqdm
from typing import Union
from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch import nn
import torch.distributed as dist
from utils import to_numpy, printlog, save_qualitative_results
from .VitMae_Manager import VitMaeManager

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


class MultiVitMaeManager(VitMaeManager, ABC):
    """manager to train MultiVitMae model with alternating training across modalities i.e one modality per step"""

    @property
    def sequence_length(self) -> dict:
        modalities = list(self.flat_model.backbone.encoder.patch_embeders.keys())
        return {modality: self.flat_model.backbone.encoder.patch_embeders[modality].num_patches
                for modality in modalities}

    def get_current_sequence_length(self, modality: str):
        return self.sequence_length[modality]

    def forward_train_step(self, img: torch.Tensor, reduce_batch: bool = True, **kwrargs):
        assert 'modality' in kwrargs, f'modality must be passed in kwrargs for MultiVitMae, got {kwrargs}'
        modality = kwrargs['modality']
        with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_autocast):
            ret = self.model(img, is_training=True, modality=modality)
            # ret: dict with keys: "cls_token", "output", "mask"
            # cls_token = outs['cls_token']
            output = ret['output']
            mask_tensor = ret['mask']

            # mask_tensor: (B, L)
            # if mask is not None:
            #     mask = repeat(mask, 'b l -> b l p', p=np.prod(self.trunk.patch_embeders[modality].patch_size))
            loss = self.get_loss(img, output, mask_tensor, reduce_batch=reduce_batch)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        self.optimiser.zero_grad()
        t0 = datetime.datetime.now()
        accumulated_iter = 0
        # get current epoch's loader
        loader = self.data_loaders['train_loader'].get_loader(epoch=self.epoch + self.start_epoch)
        for batch_ind, batch in enumerate(loader):
            # prepare batch
            modality = list(batch.keys())[0]  # get current batch's modality
            img = batch[modality].data  # batch is dict[modality] = BatchSample object with attributes data, label, etc
            img = img.to(self.device, non_blocking=True)

            # forward and gradient accumulation
            grad_context = self.get_grad_context(accumulated_iter)

            losses_accum = {'loss': torch.tensor(0.0, device=self.device, dtype=self.dtype)}
            with grad_context:
                ret = self.forward_train_step(img, modality=modality)
                loss = ret['loss'] / self.grad_accumulation_steps

            losses_accum['loss'] += loss.detach()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated_iter += 1

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
                tps = (self.batch_size * self.grad_accumulation_steps * self.get_current_sequence_length(modality)) / (dt/1000)

                # logging
                self.train_logging(batch_ind, None, losses_accum, tpb=dt, tps=tps, norm=norm, modality=modality)

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

        val_loss_all = 0
        num_modalities = len(list(self.data_loaders['valid_loader'].keys()))
        with torch.no_grad():
            for modality in self.data_loaders['valid_loader'].keys():
                modality_loader = self.data_loaders['valid_loader'][modality]
                for batch_index, batch in enumerate(tqdm(modality_loader)):
                    img, metadata = batch
                    img = img.to(self.device, non_blocking=True)
                    # forward
                    ret = self.forward_train_step(img, modality=modality, reduce_batch=False)
                    loss = ret['loss']
                    valid_loss += loss.sum().item()
                    if batch_index == 2 and self.debugging:
                        break
                # mean val loss
                valid_loss /= len(self.data_loaders['valid_loader'][modality]) * \
                              self.data_loaders['valid_loader'][modality].batch_size
                self.valid_logging(valid_loss, modality=modality)

            val_loss_all += (1/num_modalities) * valid_loss
            self.valid_logging(valid_loss=None, val_loss_all=val_loss_all)

    def train_logging(self, batch_num, data, loss, tpb, tps, norm, **kwargs):

        assert 'modality' in kwargs, f'modality must be passed in kwrargs for MultiVitMae.train_logging, got {kwargs}'
        modality_current = kwargs['modality']  # this step's modality

        # train logging
        if self.scheduler is not None and self.config['train']['lr_batchwise']:
            lr = self.scheduler.get_lr()[0]

        info_string = f"{modality_current} "
        if (self.rank == 0) and self.use_wandb and (self.global_step % self.config["logging"]['wandb_step'] == 0):
            wandb.log({"loss": loss.item(), "lr": lr, "t_per_batch": tpb}, step=self.global_step)
            wandb.log({f"loss_m_{modality_current}": loss.item()})  # save modality-specific loss

        losses_string = f"loss_m_{modality_current}: {loss['loss'].item():.4f} "

        if self.global_step % self.config["logging"]['display_step'] == 0:
            current_epoch = self.epoch + self.start_epoch
            printlog(f"Epoch {current_epoch:03d} | step {self.global_step:06d} | Batch {batch_num:03d} "
                     f"| {losses_string} | grad_norm: {norm:.4f} "
                     f"| {info_string} dt: {tpb:.1f} | tokens/sec {tps:.1f} "
                     f"r {self.rank:01d}")

        if self.global_step % self.config["logging"]["train_img_log_step"] == 0:
            train_loader = self.data_loaders[self.train_schedule[self.epoch]].get_loader(epoch=
                                                                                         self.epoch+self.start_epoch)
            # generating qualitative results for both modalities
            for loader in train_loader.dataloaders:
                modality = loader.dataset.modality
                if hasattr(loader.dataset, 'use_omnivision_api'):
                    loader.dataset.use_omnivision_api = False  # turn off ominvision api

                temp = loader.dataset.return_metadata
                loader.dataset.return_metadata = False  # turn off metadata
                save_qualitative_results(loader=loader,
                                         num_images=self.config['logging']['max_valid_imgs'],
                                         manager=self,
                                         filename=f'visuals_train_{modality}_@{self.global_step}.png',
                                         tag='_multimae')

                if hasattr(loader.dataset, 'use_omnivision_api'):
                    loader.dataset.use_omnivision_api = True  # turn on ominvision api
                loader.dataset.return_metadata = temp   # restore return_metadata

    def valid_logging(self, valid_loss, **kwargs):
        """
        logging - checkpoint saving - best val tracking
        Controlled by config using the following entries
             "logging": {
             "valid_freq": 1,
             "checkpoint_epoch": 25,
             "wandb": true,
             "wandb_step": 50,
             "display_step": 1,
             "train_img_log_step": 2500,
             "val_image_log_epoch": 10,
             "max_valid_imgs": 8
         }
        """
        self.metrics['final_epoch_step'] = [self.epoch + self.start_epoch, self.global_step - 1]

        if 'val_loss_all' not in kwargs:
            # logs of a single modality
            assert 'modality' in kwargs, f'modality must be in kwargs for MultiVitMae.train_logging, got {kwargs}'
            modality_current = kwargs['modality']  # this step's modality
            self.metrics[f'final_val_loss_{modality_current}'] = valid_loss
            self.metrics['final_epoch_step'] = [self.epoch + self.start_epoch, self.global_step - 1]
            if self.epoch % self.config['logging']['val_image_log_epoch'] == 0:
                save_qualitative_results(self.data_loaders['valid_loader'][modality_current],
                                         self.config['logging']['max_valid_imgs'],
                                         self,
                                         f'visuals_val_{modality_current}_@{self.global_step}.png',
                                         tag='_multimae')

            if (self.rank == 0) and self.use_wandb:
                # wandb.log({"val_loss": valid_loss}, step=self.global_step-1)
                wandb.log({f"val_loss_{modality_current}": valid_loss}, step=self.global_step-1)

        else:
            self.metrics['final_val_loss'] = valid_loss
            # log loss across all modalities and checkpointing
            valid_loss = kwargs['val_loss_all']
            if valid_loss < self.best_loss:
                prev_best_loss = self.best_loss
                self.best_loss = valid_loss
                self.metrics.update({'best_val_loss': valid_loss,
                                     'best_loss_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
                printlog("New best validation loss: {:.5f} prev {:.5f}".format(self.best_loss, prev_best_loss))
                self.save_checkpoint(save_as='best')

            elif (self.epoch % self.config['logging']['checkpoint_epoch'] == 0) \
                    and self.epoch > 0 or \
                    ((self.epoch + self.start_epoch) == self.config['train']['epochs'] - 1):
                self.save_checkpoint()

                if (self.rank == 0) and self.use_wandb:
                    wandb.log({"val_loss": valid_loss}, step=self.global_step - 1)

        self.write_info_json()
