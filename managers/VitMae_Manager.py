import datetime
from contextlib import nullcontext
from tqdm import tqdm
from typing import Union
from abc import ABC
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch import nn
from einops import rearrange
from managers.BaseManager import BaseManager
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


class VitMaeManager(BaseManager, ABC):
    @property
    def backbone(self):
        if isinstance(self.model, ParallelType):
            return self.model.module.backbone
        else:
            return self.model.backbone

    @property
    def sequence_length(self):
        return self.flat_model.backbone.encoder.patch_embed.num_patches

    @property
    def patch_shape(self):
        backbone = self.backbone
        assert hasattr(backbone, 'encoder') or hasattr(backbone, 'encoders'), f"backbone has no encoder or encoders"
        if not hasattr(backbone, 'encoder'):
            # for MultiModalMAE/MultiVitMaeCM objects
            if hasattr(backbone, 'encoders'):
                some_modality = list(backbone.encoders.keys())[0]
                # note: we assume all modalities have the same patch size
                return backbone.encoders[some_modality].patch_embed.patch_size

        elif hasattr(backbone.encoder, 'patch_embed'):
            return backbone.encoder.patch_embed.patch_size
        elif hasattr(backbone.encoder, 'patch_embeders'):
            some_modality = list(backbone.encoder.patch_embeders.keys())[0]
            return backbone.encoder.patch_embeders[some_modality].patch_size
        else:
            raise ValueError('Could not find patch_embed or patch_embeders in the current model')

    @staticmethod
    def patchify_img(img: torch.Tensor, patch_shape: Union[list, tuple]) -> torch.Tensor:
        """
         img: (B, C, H, W)
         patch_shape: (p, p)
         :return x (B, L, p^2 * C) with L =  H * W / (p^2)  (e.x for p = 16 C=1 B,196,256, for C=2 B,196,768)
        """
        assert isinstance(patch_shape, list) or isinstance(patch_shape, tuple), 'patch_shape must be a list or tuple'
        assert len(patch_shape) == 2, f'patch_shape must be of length 2 instead got {patch_shape}'
        assert patch_shape[0] == patch_shape[1], f'patch_shape must be square instead got {patch_shape}'
        p = patch_shape[0]
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    @staticmethod
    def unpatchify_img(img: torch.Tensor, H: int, W: int, C: int, patch_shape: Union[list, tuple]) -> torch.Tensor:
        """
        img: (B, L, p^2 * C) with L = H * W / (p^2)
        patch_shape: (p, p)
        :return tensor (B, C, H, W)
        """
        assert isinstance(patch_shape, list) or isinstance(patch_shape, tuple), 'patch_shape must be a list or tuple'
        assert len(patch_shape) == 2, f'patch_shape must be of length 2 instead got {patch_shape}'
        assert patch_shape[0] == patch_shape[1], f'patch_shape must be square instead got {patch_shape}'
        L = img.shape[1]
        p = patch_shape[0]
        h = H // p
        w = W // p
        assert L == h * w, f'L must be equal to h*w got {L} but h*w = {h} * {w} = {h*w}'
        x = rearrange(img, 'b (h w) (p1 p2 c) -> b h w p1 p2 c', h=h, w=w, p1=p, p2=p, c=C)
        x = rearrange(x, 'b h w p1 p2 c -> b c h p1 w p2')
        x = rearrange(x, 'b c h p1 w p2 -> b c (h p1) (w p2)')
        return x

    @staticmethod
    def patchify_video(img: torch.Tensor, patch_shape: Union[list, tuple]) -> torch.Tensor:
        """
        img: (B, C, T, H, W), 2D images are a special case with T =1
        x: (B, L, P_t * P^2 * C) with L = T * H * W / (P_t * P^2)
        :return x (B, L, P_t* P^2 * C)
        """
        assert isinstance(patch_shape, list) or isinstance(patch_shape, tuple), 'patch_shape must be a list or tuple'
        assert len(patch_shape) == 3, f'patch_shape must be of length 3 instead got {patch_shape}'
        p_t = patch_shape[0]
        assert patch_shape[1] == patch_shape[2], f'patch must be square instead got {patch_shape}'
        p = patch_shape[1]
        assert img.shape[3] == img.shape[4] and img.shape[3] % p == 0 and img.shape[2] % p_t == 0, \
            f'img must be square along spatial dims (H,W) [{img.shape[-2:]}]' \
            f' which must be divisible by spatial patch size [{p}]' \
            f'and img temporal dimension (T) [{img.shape[-3]}] must be divisible by temporal patch size [{p_t}]'
        x = rearrange(img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=p_t, p1=p, p2=p)
        return x
        #
        # b = img.shape[0]
        # c = img.shape[1]
        # patch layout (t, h, w) , L = t * h * w = T * H * W / (P_t * P^2)
        #
        # # number of patches across time
        # t = img.shape[2] // p_t  # t=T//P_t
        # # number of patches across height and width
        # h = img.shape[3] // p  # h=H//P
        # w = img.shape[4] // p  # w=W//P
        # # x = img.reshape(shape=(b, c, t, p_t, -1))
        # x = img.reshape(shape=(b, c, t, p_t, h, p, w, p))  # (b, c, t, p_t, h, p, w, p)
        # # in einsum (p, q, r) stands for (p_t, p, p)
        # # n c t p h q w r -> n t h w p q r c
        # x = torch.einsum("b c t p h q w r -> b t h w p q r c", x)
        # L = t * h * w
        # x = x.reshape(shape=(b, L, p_t * p * p, c))
        # x = x.reshape(shape=(b, L, -1))
        # return x  # (B, L, P_t * P^2 * C)

    @staticmethod
    def unpatchify_video(img, H, W, T, C, patch_shape) -> torch.Tensor:
        """
        img: (B, L, P_t* P^2 * C) , 2D images are a special case with T =1
         with L = T * H * W / (P_t * P^2)
        :return tensor (B, C, T, H, W)
        """
        # patch_shape = self.model.backbone.trunk.patch_embed.patch_size  # patch_shape (p_t, p_h, p_w)
        L = img.shape[1]
        if len(patch_shape) == 2:
            # if patch_shape is (p_h, p_w) then make it (1, p_h, p_w)
            patch_shape = [1] + patch_shape

        p_t = patch_shape[0]
        p_h = patch_shape[1]
        p_w = patch_shape[2]

        assert p_h == p_w
        p = p_h
        h = H // p
        w = W // p
        t = T // p_t
        assert L == t * h * w

        # current todo figure out what was wrong with these alternative implementations
        # # x = img.reshape(shape=(img.shape[0], t, h, w, p_t, p, p, C))
        # x = img.reshape(shape=(img.shape[0], h, w, t, p, p, p_t, C))
        # x = torch.einsum("b h w t p r q c -> b c t r h q w p", x)
        # x = x.reshape(shape=(img.shape[0], C, t * p_t, h * p, w * p))

        # x = img.reshape(shape=(img.shape[0], h, w, t, p, p, p_t, C))
        # x = torch.einsum("b h w t p r q c -> b c t r h q w p", x)
        # x = x.reshape(shape=(img.shape[0], C, t * p_t, h * p, w * p))
        # img              (b, L,      P_t * P^2 * C)

        x = rearrange(img, 'b (t h w) (p0 p1 p2 c) -> b t h w p0 p1 p2 c', t=t, h=h, w=w, p0=p_t, p1=p, p2=p, c=C)
        x = rearrange(x, 'b t h w p0 p1 p2 c -> b c t p0 h p1 w p2')
        x = rearrange(x, 'b c t p0 h p1 w p2 -> b c (t p0) (h p1) (w p2)')
        return x

    @staticmethod
    def show_image(image, title=''):
        # image is (H, W, 3) or (H,W,1)
        if image.shape[2] == 1:
            image = image.squeeze(2)
        else:
            assert image.shape[2] == 3
        plt.imshow(torch.clip(image * 255, 0, 255).int())
        plt.title(title, fontsize=8)
        plt.axis('off')
        # plt.show()
        return

    def post_process_output_img(self, img, output, mask):
        """ unpatchify prediction and return the masked image, the pasted image and the prediction
        :param img: (1, C, H, W)
        :param output: (1, L, p^2 * C)
        :param mask: (1, L)
        :return: im_masked, im_paste, prediction (all (1, C, H, W))
        """
        if img.shape[0] > 1:
            img = img[0].unsqueeze(0)
            output = output[0].unsqueeze(0)
            mask = mask[0].unsqueeze(0)

        patch_size = self.patch_shape
        # ensure this is (1, C, T, H, W) by replication along the temporal axis
        assert len(img.shape) == 4
        b, C, H, W = img.shape
        prediction = self.unpatchify_img(output, H, W, C, self.patch_shape)  # y (B, C, T, H, W)
        prediction = torch.einsum('bchw->bhwc', prediction).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.view(1, -1, 1)  # (1, h, w) -> (1, L) : L = t*h*w)
        mask = mask.repeat(1, 1, (patch_size[0] ** 2) * C)  # (1, L, p^2*c)
        mask = self.unpatchify_img(mask, H, W, C, self.patch_shape)  # 1 is removing, 0 is keeping
        # only visualize first element along the time axis (T)
        # channels last
        mask = torch.einsum('bchw->bhwc', mask).detach().cpu()
        x = torch.einsum('bchw->bhwc', img).cpu()

        # masked image
        mask = 1.0 * mask
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + prediction * mask
        return im_masked, im_paste, prediction

        # make the plt figure larger
        # plt.rcParams['figure.figsize'] = [24, 24]
        #
        # plt.subplot(1, 4, 1)
        # self.show_image(x[0], "original")
        #
        # plt.subplot(1, 4, 2)
        # self.show_image(im_masked[0], "masked")
        #
        # plt.subplot(1, 4, 3)
        # self.show_image(y[0], "reconstruction")
        #
        # plt.subplot(1, 4, 4)
        # self.show_image(im_paste[0], "reconstruction + visible")
        #
        # plt.savefig(f'mae_visual_train_{self.global_step}.png')
        # return output

    def post_process_output_video(self, img, output, mask):
        """
        :param img: (1, C, T, H, W)
        :param output: (1, L, P_t * P^2 * C)
        :param mask: (1, L)
        :return: im_masked, im_paste, y
        """
        if img.shape[0] > 1:
            img = img[0].unsqueeze(0)
            output = output[0].unsqueeze(0)
            mask = mask[0].unsqueeze(0)

        # ensure this is (1, C, T, H, W) by replication along the temporal axis
        p_t = self.patch_shape[0]
        if len(img.shape) == 4:
            img = img.unsqueeze(2).repeat([1, 1, p_t, 1, 1])  # tile by temporal size of patches
        else:
            assert len(img.shape) == 5, f'img must be 4 or 5 dim, got {img.shape}'
            assert img.shape[2] % p_t == 0

        b, C, T, H, W = img.shape
        y = self.unpatchify_video(output, H, W, T, C, self.patch_shape)  # y (B, C, T, H, W)
        y = torch.einsum('bcthw->bthwc', y).detach().cpu()

        # visualize the mask
        mask = mask.detach()
        mask = mask.view(1, -1)  # (1, t, h, w) -> (1, L) : L = t*h*w)
        # (1, L, p_t*p^2*c)
        mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_shape[0] * (self.patch_shape[1] ** 2) * C)
        mask = self.unpatchify_video(mask, H, W, T, C, self.patch_shape)  # 1 is removing, 0 is keeping

        # only visualize first element along the time axis (T)
        mask = torch.einsum('bcthw->bthwc', mask).detach().cpu()
        x = torch.einsum('bcthw->bthwc', img).cpu()

        # masked image
        mask = 1.0 * mask
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask
        return im_masked, im_paste, y

    def get_grad_context(self, accumulated_iter):
        # if we are not using DDP, then we a call of backward is always
        if self.parallel:
            return nullcontext()

        # if we are accumulating gradients, we only sync at the last accumulation step
        sync_gradients = (accumulated_iter + 1) % self.grad_accumulation_steps == 0
        # if the model is DDP, but we are not at the last accumulation step then we use no_sync as context
        # otherwise we use the default context
        if (not sync_gradients) and self.parallel:
            grad_context = self.model.no_sync()
        else:
            grad_context = nullcontext()

        if self.debugging: printlog(f"grad_contex: {self.global_step}== {grad_context}")

        return grad_context

    def forward_train_step(self, img, reduce_batch=True, **kwrargs):
        ret = dict()
        b = img.shape[0]
        output, mask_tensor = self.model(img, is_training=True)
        loss = self.get_loss(img, output, mask_tensor.view(b, -1), reduce_batch=reduce_batch)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def get_loss(self, img, pred, mask, reduce_batch=True):
        """
        imgs: [N, C, T, H, W]
        pred: [N, L, P_t * P^2 * C]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if self.config['loss'].get('cotrain_with_volumes', False):
            p_t = self.model.backbone.patch_shape[0]
            if len(img.shape) == 4:
                img = img.unsqueeze(2).repeat([1, 1, p_t, 1, 1])  # tile by temporal size of patches
            else:
                assert len(img.shape) == 5, f'img must be 4 or 5 dim, got {img.shape}'
                assert img.shape[2] % p_t == 0, f'img.shape[2] {img.shape[2]} must be divisible by p_t {p_t}'

        # sanity check for patchify and unpatchify for videos
        # img (B,C,T,H,W) -> (B,L,P_t*P^2*C)
        # target = self.patchify_video(img)
        # img_rec = self.unpatchify_video(target, T=2, H=224, W=224, C=3)
        # tensor2pil_show_label_color(img_rec)

        # gif sanity check for patchify and unpatchify for videos
        # save_tensors_as_gif(img_tensors=img[0], duration=0.3, filename='original.gif')
        # save_tensors_as_gif(img_tensors=self.unpatchify_video(target, 224, 224, 16, 3, self.patch_shape)[0],
        # duration=0.3, filename='recon.gif')

        if self.config['loss'].get('cotrain_with_volumes', False):
            target = self.patchify_video(img, self.patch_shape)
        else:
            target = self.patchify_img(img, self.patch_shape)

        with torch.no_grad():
            if self.config['loss'].get('patch_normalization', False):
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5  # todo review this normalization
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        if reduce_batch:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches only
        else:
            loss = (loss * mask).sum(dim=-1) / mask.sum(dim=-1)

        return loss

    def train_one_epoch(self):
        """Train the model for one epoch"""
        self.model.train()
        self.optimiser.zero_grad()
        accumulated_iters = 0  # used if gradient accumulation is used else has no effect
        t0 = datetime.datetime.now()
        for batch_ind, batch in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
            # prepare batch
            img, metadata = batch

            img = img.to(self.device, non_blocking=True)

            # forward
            ret = self.forward_train_step(img)
            loss = ret['loss']
            output = ret['output']

            # backward
            loss.backward()
            accumulated_iters += 1
            if accumulated_iters % self.grad_accumulation_steps == 0:
                accumulated_iters = 0  # reset
                if self.debugging:
                    printlog(f"accum_iters: {self.grad_accumulation_steps} global_step: {self.global_step}")
                # update model
                self.optimiser.step()
                self.optimiser.zero_grad()

                # lr scheduler step
                if self.scheduler is not None and self.config['train']['lr_batchwise']:
                    self.scheduler.step()
                dt = (datetime.datetime.now() - t0).total_seconds() * 1000
                t0 = datetime.datetime.now()

                # logging
                self.train_logging(batch_ind, output, img, loss, dt)

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

        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
                img, metadata = batch
                img = img.to(self.device, non_blocking=True)

                # forward
                ret = self.forward_train_step(img, reduce_batch=False)
                loss = ret['loss']
                valid_loss += loss.sum().item()
                if batch_index == 2 and self.debugging:
                    break
            # mean val loss
            valid_loss /= (len(self.data_loaders['valid_loader']) * self.valid_batch_size)
            # valid_loss /= len(self.data_loaders['valid_loader'])
            self.valid_logging(valid_loss)

    def train_logging(self, batch_num, data, loss, b, tpb, tps, norm, **kwargs):
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

        # train logging
        if self.scheduler is not None and self.config['train']['lr_batchwise']:
            lr = self.scheduler.get_lr()[0]

        info_string = " "
        if (self.rank == 0) and self.use_wandb and (self.global_step % self.config["logging"]['wandb_step'] == 0):
            wandb.log({"loss": loss.item(), "lr": lr, "t_per_batch": b}, step=self.global_step)

        if self.global_step % self.config["logging"]['display_step'] == 0:
            printlog("Epoch {:03d} iter {:06d}, Batch {:03d} - Loss: {:.4f}; {} t: {:.1f} r {} ".format(
                self.epoch + self.start_epoch, self.global_step, batch_num, loss.item(), info_string, tpb, self.rank))

        if self.global_step % self.config["logging"]["train_img_log_step"] == 0:

            if self.config['loss'].get('cotrain_with_volumes', False):
                # self.post_process_output_video(img, output, mask_tensor)
                save_qualitative_results(self.data_loaders[self.train_schedule[self.epoch]],
                                         self.config['logging']['max_valid_imgs'],
                                         self,
                                         f'visuals_train_@{self.global_step}.gif',
                                         tag='_mae_vol')
            else:
                save_qualitative_results(self.data_loaders[self.train_schedule[self.epoch]],
                                         self.config['logging']['max_valid_imgs'],
                                         self,
                                         f'visuals_train_@{self.global_step}.png',
                                         tag='_mae')

        # save_qualitative_results(loader, num_images, manager: VitMae_manager, filename)

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
            self.save_checkpoint(save_as='best')  # fixme

        elif (self.epoch % self.config['logging']['checkpoint_epoch'] == 0) \
                and self.epoch > 0 or \
                ((self.epoch + self.start_epoch) == self.config['train']['epochs'] - 1):

            self.save_checkpoint()

        if self.epoch % self.config['logging']['val_image_log_epoch'] == 0:
            if self.config['loss'].get('cotrain_with_volumes', False):
                # self.post_process_output_video(img, output, mask_tensor)
                save_qualitative_results(self.data_loaders['valid_loader'],
                                         self.config['logging']['max_valid_imgs'],
                                         self,
                                         f'visuals_val_@{self.global_step}.gif',
                                         tag='_mae_vol')
            else:
                save_qualitative_results(self.data_loaders['valid_loader'],
                                         self.config['logging']['max_valid_imgs'],
                                         self,
                                         f'visuals_val_@{self.global_step}.png',
                                         tag='_mae')

        if (self.rank == 0) and self.use_wandb:
            wandb.log({"val_loss": valid_loss}, step=self.global_step-1)

        self.write_info_json()
