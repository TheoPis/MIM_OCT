import datetime
from tqdm import tqdm
from typing import Union, Dict
from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch import nn
from managers.BaseManager import BaseManager
# from managers.MultiVitSeq_Manager import MultiVitSeqManager
from utils import to_numpy, printlog, save_qualitative_results, save_tensors_as_gif
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


class CLIPManager(BaseManager, ABC):

    def post_process_output_img(self, img, output, mask):
        pass

    def post_process_output_video(self, img, output, mask):
        pass

    def forward_train_step(self, data: Dict[str, torch.Tensor], single_rank=False, **kwrargs):
        """ clip training using 2 visual modalities
        -data: a dict with keys the names of modalities: eg "OCT", "FA" and values the tensors of the batch
        """
        assert isinstance(data, dict), f'input data must be a dict, got {type(data)} with keys modality names'
        # we assume two modalities todo generalize
        ret = dict()
        modalities = list(data.keys())
        m1 = modalities[0]
        m2 = modalities[1]
        feats_m1, feats_m2, scale = self.model(data[m1], data[m2])
        # compute loss only using rank=0 process else all ranks
        # world_size = self.loss.world_size
        if single_rank:
            self.loss.world_size = 1

        loss = self.loss(feats_m1, feats_m2, scale)

        # # compute loss only using rank=0 process
        # self.loss.world_size = world_size

        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        a = datetime.datetime.now()
        for batch_index, batch in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
            data = batch
            modalities = batch.keys()

            b = (datetime.datetime.now() - a).total_seconds() * 1000
            a = datetime.datetime.now()
            for modality in modalities:
                data[modality] = data[modality].to(self.device, non_blocking=True)

            self.optimiser.zero_grad()

            # forward
            ret = self.forward_train_step(data)
            loss = ret['loss']

            # backward
            loss.backward()
            self.optimiser.step()

            # lr scheduler
            if self.scheduler is not None and self.config['train']['lr_batchwise']:
                self.scheduler.step()

            # logging
            #     def train_logging(self, batch_num, data,  tpb, losses_dict: Dict[str, torch.Tensor], **kwargs):
            self.train_logging(batch_index, data, tpb=b, losses_dict=ret, modality=modalities)

            # increment device-wise step
            self.global_step += 1

            if batch_index == 10 and self.debugging:
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

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
                data = batch
                modalities = batch.keys()

                for modality in modalities:
                    data[modality] = data[modality].to(self.device, non_blocking=True)

                ret = self.forward_train_step(data, single_rank=True)
                loss = ret['loss']
                valid_loss += loss.sum().item()
                if batch_index == 2 and self.debugging:
                    break
            # mean val loss
            valid_loss /= len(self.data_loaders['valid_loader']) * self.config['data']['valid_batch_size']
            self.valid_logging(valid_loss)

    def train_logging(self, batch_num, data, tpb, losses_dict: Dict[str, torch.Tensor], **kwargs):
        """
        #   "logging": {
        #     "valid_freq": 1,
        #     "checkpoint_epoch": 25,
        #     "wandb": true,
        #     "wandb_step": 50,
        #     "display_step": 1,
        #     "train_img_log_step": 2500,
        #     "max_valid_imgs": 8
        #   }
        # }
        """
        modalities_current = list(data.keys())
        # train logging
        if self.scheduler is not None and self.config['train']['lr_batchwise']:
            lr = self.scheduler.get_lr()[0]

        info_string = ""
        for modality in modalities_current:
            info_string += f"{modality} "

        if (self.rank == 0) and self.use_wandb and (self.global_step % self.config["logging"]['wandb_step'] == 0):
            wandb.log({"loss": losses_dict['loss'].item(), "lr": lr, "t_per_batch": tpb}, step=self.global_step)
            # for m in modalities_current:
            #     wandb.log({f"loss_m_{m}": losses_dict[f'loss_m_{m}'].item()})  # save modality-specific loss
        if self.global_step % self.config["logging"]['display_step'] == 0:
            printlog("Epoch {:03d} iter {:06d}, Batch {:03d} - Loss: {:.4f}; {} t: {:.1f} r {} ".format(
                self.epoch + self.start_epoch, self.global_step, batch_num, losses_dict['loss'].item(),
                info_string, tpb, self.rank))

        if self.global_step % self.config["logging"]["train_img_log_step"] == 0:
            pass

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
        self.metrics['final_val_loss'] = valid_loss
        self.metrics['final_epoch_step'] = [self.epoch + self.start_epoch, self.global_step - 1]

        if valid_loss < self.best_loss:
            prev_best_loss = self.best_loss
            self.best_loss = valid_loss
            self.metrics.update({'best_val_loss': valid_loss,
                                 'best_loss_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            printlog("New best validation loss: {:.5f} prev {:.5f}".format(self.best_loss, prev_best_loss))
            if self.epoch > 0:
                self.save_checkpoint(save_as='best')

        elif (self.epoch % self.config['logging']['checkpoint_epoch'] == 0) \
                and self.epoch > 0 or \
                ((self.epoch + self.start_epoch) == self.config['train']['epochs'] - 1):

            self.save_checkpoint()

        if self.epoch % self.config['logging']['val_image_log_epoch'] == 0:
            # no visual for clip
            pass

        if (self.rank == 0) and self.use_wandb:
            wandb.log({"val_loss": valid_loss}, step=self.global_step-1)

        self.write_info_json()
