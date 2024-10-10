import datetime
import wandb
import os
import pathlib
import cv2
from abc import ABC
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
from torch.nn import functional as F

from managers.BaseManager import BaseManager
from losses import LossWrapper
from utils import mask_to_colormap, get_remapped_colormap, remap_mask, create_new_directory, reverse_mapping,\
    t_get_confusion_matrix, t_normalise_confusion_matrix, t_get_pixel_accuracy, get_matrix_fig, to_numpy,\
    t_get_mean_iou, DATASETS_INFO, printlog, shorten_key


class OCRNetManager(BaseManager, ABC):
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

    def forward_train_step(self, img, lbl, **kwargs):
        ret = dict()

        # todo find a cleaner way to pass modality: if is_multimodal then pass modality
        #  self.model can be ddp so self.model.module needs to be accessed
        if hasattr(self.flat_model, 'is_multimodal'):
            is_multimodal = self.flat_model.is_multimodal
        else:
            is_multimodal = False

        ########################################
        with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_autocast):
            if is_multimodal:
                output = self.model(img.float(), modality=self.model.modality)
            else:
                output = self.model(img.float())
            loss = self.loss(output, lbl.long())
        ########################################

        # get separate loss terms values for logging (stored in self.loss during loss computation)
        if 'individual_losses' in kwargs:
            individual_losses = kwargs['individual_losses']
            for key in self.loss.loss_vals:
                individual_losses[key] += self.loss.loss_vals[key]
            ret['individual_losses'] = individual_losses

        ret['output'] = output
        ret['loss'] = loss

        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def forward_val_step(self, img, lbl, skip_loss=False, **kwargs):
        ret = dict()
        if hasattr(self.flat_model, 'is_multimodal'):
            is_multimodal = self.flat_model.is_multimodal
        else:
            is_multimodal = False
        ########################################
        if is_multimodal:
            output = self.model(img.float(), modality=self.model.modality)
        else:
            output = self.model(img.float())
        loss = None
        if not skip_loss:
            loss = self.loss(output, lbl.long())
            if 'individual_losses' in kwargs:
                individual_losses = kwargs['individual_losses']
                for key in self.loss.loss_vals:
                    individual_losses[key] += self.loss.loss_vals[key]
                ret['individual_losses'] = individual_losses
        ########################################
        ret['output'] = output
        ret['loss'] = loss

        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def post_process_output(self, img, output, lbl, metadata, skip_label=False):
        if metadata and self.dataset in ['PASCALC', 'ADE20K']:
            if "pw_ph_stride" in metadata:
                # undo padding due to fit_stride resizing
                pad_w, pad_h, stride = metadata["pw_ph_stride"]
                if pad_h > 0 or pad_w > 0:
                    output = output[:, :, 0:output.size(2) - pad_h, 0:output.size(3) - pad_w]
                    lbl = lbl[:, 0:output.size(2) - pad_h, 0:output.size(3) - pad_w]
                    img = img[:, :, 0:output.size(2) - pad_h, 0:output.size(3) - pad_w]

            if "sh_sw_in_out" in metadata:
                if hasattr(self.model, 'module'):
                    align_corners = self.model.module.align_corners
                else:
                    align_corners = self.model.align_corners
                # undo resizing
                starting_size = metadata["sh_sw_in_out"][-2]
                # starting size is w,h thanks PIL
                output = F.interpolate(input=output, size=starting_size[-2:][::-1],
                                       mode='bilinear', align_corners=align_corners)
                img = F.interpolate(input=img, size=starting_size[-2:][::-1],
                                    mode='bilinear', align_corners=align_corners)
                lbl = metadata["original_labels"].squeeze(0).long().cuda()

        return img, output, lbl

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
                if self.debugging and batch_ind == 2:
                    print(f'stopping at {batch_ind}')
                    break

    @torch.no_grad()
    def validate(self):
        """Validate the model on the validation data"""
        if not (self.rank == 0):
            return
        if not self.parallel:
            torch.backends.cudnn.benchmark = False
        self.model.eval()
        valid_loss = 0
        confusion_matrix = None
        individual_losses = dict()
        if isinstance(self.loss, LossWrapper):
            for key in self.loss.loss_vals:
                individual_losses[key] = 0

        for batch_ind, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
            batch_dict = self.unpack_batch(batch)
            img, lbl, metadata = batch_dict['img'], batch_dict['lbl'], batch_dict['metadata']
            img = img.to(self.device, non_blocking=True)
            lbl = lbl.to(self.device, non_blocking=True)

            # forward
            ret = self.forward_val_step(img, lbl, individual_losses=individual_losses)
            loss = ret['loss']
            pred = ret['output']
            valid_loss += loss
            img, pred, lbl = self.post_process_output(img, pred, lbl, metadata)

            # logging
            confusion_matrix = t_get_confusion_matrix(pred, lbl, self.dataset, confusion_matrix)
            individual_losses = ret['individual_losses'] if 'individual_losses' in ret else individual_losses
            if self.debugging and batch_ind == 2:
                print(f'stopping at {batch_ind}')
                break
        valid_loss /= len(self.data_loaders['valid_loader'])
        pa, pac = t_get_pixel_accuracy(confusion_matrix)
        mious = t_get_mean_iou(confusion_matrix, self.config['data']['experiment'], self.dataset, True, rare=True)
        # logging + checkpoint
        self.valid_logging(valid_loss, confusion_matrix, individual_losses, mious, pa, pac)
        if not self.parallel:
            torch.backends.cudnn.benchmark = True
        return mious

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
        info_string = " "  # info that will be printed to console
        wandb_dict = {}  # dictionary to be logged to wandb
        lr = self.scheduler.get_last_lr()[-1] if self.scheduler is not None else self.config['train']['learning_rate']

        # basic info
        if self.use_wandb and (self.global_step % self.config["logging"]["wandb_step"] == 0) and (self.rank == 0):
            wandb_dict.update({"loss": loss, "lr": lr, "t_per_batch": dt, 'norm': norm})
            # log each log term separately
            if hasattr(self.loss, 'loss_vals'):
                dict_extra = {}
                for key in self.loss.loss_vals:
                    info_string += ' {} {:.5f}; '.format(str(shorten_key(key, is_cs=False)),
                                                         self.loss.loss_vals[key].item())
                    dict_extra = {f"{str(key)}": self.loss.loss_vals[key].item()}
                if len(dict_extra) > 0:
                    wandb_dict.update(dict_extra)
            wandb.log(wandb_dict, step=self.global_step)

        if self.global_step % self.config["logging"]['display_step'] == 0:
            current_epoch = self.epoch + self.start_epoch
            printlog(f"Epoch {current_epoch:03d} | step {self.global_step:06d} | Batch {batch_num:03d} "
                     f"| Loss: {loss:.4f} | grad_norm: {norm:.4f} | {info_string} dt: {dt:.1f} | tokens/sec {tps:.1f} "
                     f"r {self.rank:01d}")

        # todo this causes freezes in the ddp case
        # # wandb images logging
        # if self.global_step % self.config["logging"]["train_img_log_step"] == 0 and (self.rank == 0):
        #     filename = save_qualitative_results(self.data_loaders[self.train_schedule[self.epoch]],
        #                                         self.config['logging']['max_valid_imgs'],
        #                                         self,
        #                                         f'visuals_train_@{self.global_step}.png',
        #                                         tag='_seg',
        #                                         use_wandb=self.use_wandb,
        #                                         wandb_tag='train')
        # if self.use_wandb:
        #     wandb.log({"train_visuals": wandb.Image(filename)}, step=self.global_step)

    def valid_logging(self, valid_loss, confusion_matrix, individual_losses, mious, pa, pac):
        """ logging - checkpoint saving - best val tracking """

        wandb_dict = {"val_loss": valid_loss, "val_pac": pac, "val_pa": pa, "val_miou": mious['mean_iou']}

        info_string = ''
        if hasattr(self.loss, 'loss_vals'):
            for key in self.loss.loss_vals:
                individual_losses[key] /= len(self.data_loaders['valid_loader'])
                info_string += ' {} {:.5f}; '.format(str(key), individual_losses[key].item())
                wandb_dict.update({f"val_{str(key)}": individual_losses[key].item()})

        if (self.rank == 0) and self.use_wandb:
            wandb.log(wandb_dict, step=self.global_step - 1)
        # log confusion matrix
        row_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'row')
        col_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'col')
        fig_cm_1 = get_matrix_fig(to_numpy(row_confusion_matrix), self.config['data']['experiment'], self.dataset)
        fig_cm_2 = get_matrix_fig(to_numpy(col_confusion_matrix), self.config['data']['experiment'], self.dataset)
        f1 = os.path.join(self.log_dir, f"{'conf_matrix_row'}@{self.global_step}.png")
        f2 = os.path.join(self.log_dir, f"{'conf_matrix_col'}@{self.global_step}.png")
        printlog(f"Saving cm as {f1} and {f2}")
        fig_cm_1.savefig(f1)
        fig_cm_2.savefig(f2)

        # if self.use_wandb:
        #     images = wandb.Image(f1, caption=f"val_conf_matrix_row@{self.global_step}")
        #     wandb.log({"conf_matrix_row": images})
        #     images = wandb.Image(f2, caption=f"val_conf_matrix_col@{self.global_step}")
        #     wandb.log({"conf_matrix_col": images})

        # todo this causes freezes in the ddp case
        # if self.epoch % self.config['logging'].get('valid_img_log_epoch', 1) == 0:
        #     f3 = save_qualitative_results(self.data_loaders['valid_loader'],
        #                                   self.config['logging']['max_valid_imgs'],
        #                                   self,
        #                                   f'visuals_val_@{self.global_step}.png', tag='_seg',
        #                                   use_wandb=self.use_wandb, wandb_tag='val')
        #     if self.use_wandb:
        #         wandb.log({"val_visuals": wandb.Image(f3)}, step=self.global_step - 1)

        # console logging and model checkpoint
        msg_str = "Epoch {:03d} - val loss: {:.5f} - miou:{:.2f} ".format(self.epoch + self.start_epoch,
                                                                          valid_loss,
                                                                          mious['mean_iou'])
        categ_mious = []
        mious_values = dict()
        for categ in mious['categories']:
            categ_mious.append(mious['categories'][categ])
            # self.valid_writer.add_scalar('metrics/{}'.format(categ), categ_mious[-1], self.global_step)
            msg_str += "- {}:{:.2f}".format(categ, categ_mious[-1])
            mious_values[categ] = round(float(categ_mious[-1].cpu().numpy()), 4)

        printlog(msg_str)
        m_iou = round(float(mious['mean_iou'].cpu().numpy()), 4)

        best_miou_flag = False
        if m_iou > self.metrics['best_miou']:
            best_miou_flag = True
            self.metrics.update({'best_miou': m_iou,
                                 'best_miou_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            msg_str = "            New best mIoU (tot "
            msg1, msg2 = "", "{:.4f} ".format(m_iou)
            for categ in mious['categories']:
                self.metrics.update({'best_miou_{}'.format(categ): mious_values[categ]})
                msg1 += "/ {} ".format(categ)
                msg2 += "{:.4f} ".format(mious_values[categ])
            printlog(msg_str + msg1 + ' ): ' + msg2)

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.metrics.update({'best_loss_miou': m_iou,
                                 'best_loss_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            printlog("            New best validation loss: {:.5f}".format(valid_loss))
            msg_stra = "         --- with mIoU (tot "
            msg_strb = "         --- best mIoU (tot "
            msg1, msg2 = "", "{:.4f} ".format(m_iou)
            msg3, msg4 = "", "{:.4f} ".format(self.metrics['best_miou'])
            for categ in mious['categories']:
                msg1 += "/ {} ".format(categ)
                msg2 += "{:.4f} ".format(mious_values[categ])
                msg3 += "/ {} ".format(categ)
                self.metrics.update({'best_loss_miou_{}'.format(categ): mious_values[categ]})
                msg4 += "{:.4f} ".format(self.metrics['best_miou_{}'.format(categ)])
            if not best_miou_flag:
                printlog(msg_stra + msg1 + ' ): ' + msg2)
                printlog(msg_strb + msg3 + ' ): ' + msg4)

        # checkpoint
        if best_miou_flag:
            self.save_checkpoint(save_as='best')
        if self.epoch % self.config['log_every_n_epochs'] == 0 and self.epoch > 0\
                or (self.epoch + self.start_epoch) == self.config['train']['epochs'] - 1:
            self.save_checkpoint()

        # Update info.json file so it exists in case the run stops / crashes before self.finalise()
        for categ in mious['categories']:
            self.metrics.update({f'final_miou_{categ}': mious_values[categ]})

        self.metrics['final_miou'] = m_iou
        self.metrics['final_miou_epoch_step'] = [self.epoch + self.start_epoch, self.global_step - 1]
        self.write_info_json()

    @torch.no_grad()
    def infer(self):
        """inference for segmentation datasets only """
        assert 'load_checkpoint' in self.config, 'load_checkpoint: "run_id" must be in config for inference mode!'
        """run the model on validation data of a split , creates a logfile named 'infer' in logging dir """
        self.model.eval()
        tta, json_tag = '', ''
        confusion_matrix = None
        for batch_ind, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
            batch_dict = self.unpack_batch(batch)
            img, lbl, metadata = batch_dict['img'], batch_dict['lbl'], batch_dict['metadata']
            img = img.to(self.device, non_blocking=True)
            lbl = lbl.to(self.device, non_blocking=True)

            # forward
            output = self.model(img.float())

            # confusion matrix
            confusion_matrix = t_get_confusion_matrix(output, lbl, self.dataset, confusion_matrix)

            if self.save_outputs:
                self.save_output_seg(img, output, lbl, metadata)
                print("\r saved {}".format(batch_ind), end='', flush=True)

            if batch_ind == 30 and self.debugging:
                print(f'stopping at {batch_ind}')
                break

        mious, latex_line = self.get_and_show_seg_metrics(confusion_matrix)
        return mious, mious, latex_line  # fixme this is just to mirror what Vit_Manager.infer returns

    def get_and_show_seg_metrics(self, confusion_matrix):
        # get metrics
        mious = t_get_mean_iou(confusion_matrix, self.config['data']['experiment'], self.dataset, True, rare=True)
        self.metrics.update({f'final_miou_{categ}': mious['categories'][categ] for categ in mious['categories'].keys()})
        self.metrics['final_miou'] = mious['mean_iou']
        split = self.config['data']['split']
        if isinstance(split, list):
            split = split[-1]
        else:
            split = 'val'
        self.write_dict_json(config=mious, filename=f"{self.date}_infer_split_{split}")

        # logging
        msg_str = "\rmiou:{:.4f} ".format(mious['mean_iou'])
        for categ in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2]:
            msg_str += "- {}:{:.4f}".format(categ, to_numpy(mious['categories'][categ]))
        printlog(msg_str)
        printlog(f"iou per class: {mious['per_class_iou']}")
        latex_line = self.get_latex_line(mious, None)
        printlog(latex_line)
        return mious, latex_line

    @torch.no_grad()
    def submission_infer(self, **kwargs):
        """inference without labels for server evaluation"""
        assert 'load_checkpoint' in self.config, 'load_checkpoint: "run_id" must be in config for inference mode!'
        """run the model on validation data of a split , creates a logfile named 'infer' in logging dir """
        self.model.eval()
        for batch_ind, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
            print("\r Inference on {}".format(batch_ind), end='', flush=True)
            batch_dict = self.unpack_batch(batch)
            img, _, metadata = batch_dict['img'], batch_dict['lbl'], batch_dict['metadata']
            img = img.to(self.device, non_blocking=True)
            output = self.model(img.float())
            if self.save_outputs:
                self.save_output_seg(img, output, None, metadata)
                print("\r saved {}".format(batch_ind), end='', flush=True)
            if batch_ind == 10 and self.debugging:
                print(f'stopping at {batch_ind}')
                break

    def save_output_seg(self, img, output, lbl, metadata):
        filename = metadata['target_filename'][0]
        if 'subject_name' in metadata:
            filename = pathlib.Path(filename)
            stem = str(metadata['subject_name'][0]) + '_' + str(filename.stem) + filename.suffix
            filename = str(filename.parent / stem)

        pred = torch.argmax(nn.Softmax2d()(output), dim=1)  # contains train_ids need class_ids for evaluation
        split = 'val'
        # override split for submission_inference of RETOUCH to avoid having to set in config
        if self.dataset == 'RETOUCH' and self.config['mode'] == 'submission_inference':
            split = 'test'
        create_new_directory(str(pathlib.Path(self.log_dir) / 'outputs' / split / 'debug'))
        create_new_directory(str(pathlib.Path(self.log_dir) / 'outputs' / split / 'submit'))
        create_new_directory(str(pathlib.Path(self.log_dir) / 'outputs' / split / 'overlay'))

        debug_pred = mask_to_colormap(to_numpy(pred)[0],
                                      get_remapped_colormap(
                                          DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0],
                                          self.dataset),
                                      from_network=True, experiment=self.experiment,
                                      dataset=self.dataset)[..., ::-1]  # ::-1 this is because cv2 expects BGR
        if lbl is not None:
            debug_lbl = mask_to_colormap(to_numpy(lbl)[0],
                                         get_remapped_colormap(
                                             DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0],
                                             self.dataset),
                                         from_network=True, experiment=self.experiment,
                                         dataset=self.dataset)[..., ::-1]

        cv2.imwrite(
            str(pathlib.Path(self.log_dir) / 'outputs' / split / 'debug' / pathlib.Path(filename).stem) + '.png',
            debug_pred)

        class_to_train_mapping = DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0]
        train_to_class_mapping = reverse_mapping(class_to_train_mapping)
        submission_pred = remap_mask(to_numpy(pred)[0], train_to_class_mapping)

        cv2.imwrite(
            str(pathlib.Path(self.log_dir) / 'outputs' / split / 'submit' / pathlib.Path(filename).stem) + '.png',
            submission_pred)
        if 'torchvision_normalise' in self.config['data']['transforms_val']:
            # undo torchvision_normalise
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img.device) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img.device)

        if lbl is not None:
            if debug_pred.sum() > 0:
                # this was looking ok for retouch, aroi, OCT5K may need to be adjusted for other datasets
                # overlay_pred = cv2.addWeighted(to_numpy(255 * img)[0].transpose(1, 2, 0), 0.9, debug_pred, 0.9, 0, dtype=cv2.CV_32F)
                # debug_pred[debug_pred == 0] = 255
                overlay_pred = cv2.addWeighted(to_numpy(255 * img)[0].transpose(1, 2, 0), 0.5, debug_pred, 0.75, 0, dtype=cv2.CV_32F)
                overlay_lbl = cv2.addWeighted(to_numpy(255 * img)[0].transpose(1, 2, 0), 0.5, debug_lbl, 0.75, 0, dtype=cv2.CV_32F)
                # concat horrizontally
                overlay = np.concatenate([overlay_pred, overlay_lbl], axis=1)
                cv2.imwrite(str(pathlib.Path(self.log_dir) / 'outputs' / split / 'overlay' / pathlib.Path(filename).stem) + '.png', overlay)
