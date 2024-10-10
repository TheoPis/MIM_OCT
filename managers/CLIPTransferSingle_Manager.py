from typing import Union
from abc import ABC
import matplotlib.pyplot as plt
import torch
from torch import nn
from managers.MultiVit_Manager import MultiVitManager
plt.gray()

ParallelType = nn.parallel.DistributedDataParallel
Model = Union[nn.Module, ParallelType]


class CLIPTransferSingleManager(MultiVitManager, ABC):
    """a really not flexible wrapper for finetuning/lprobing CLIP models with a single modality downstream
       it assumes modalites are OCT + some other modality.
       in config and model, the OCT modality should be always appearing first
       during pretraining the model uses the backbone named 'visual_a' to process OCT images
       during finetuning the model uses the backbone named 'visual_a' to process OCT images
    """

    def forward_train_step(self, img, lbl, reduce_batch=True, **kwrargs):
        """
        :param img: [B, C, H, W] tensor
        :param lbl: [B, num_classes] tensor
        :param reduce_batch: whether to reduce the batch dimension in the loss
        :return:
        """
        ret = dict()

        if self.modality == 'OCT':
            output = self.model(image_a=img)
        else:
            output = self.model(image_b=img)
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
        # todo add for single modality finetuning

        if self.use_ema and self.ema_model is not None:
            if self.modality == 'OCT':
                output = self.ema_model(image_a=img)
            else:
                output = self.ema_model(image_b=img)

        else:
            if self.modality == 'OCT':
                output = self.model(image_a=img)
            else:
                output = self.model(image_b=img)

        loss = None
        if not skip_loss:
            loss = self.get_loss(img, output, lbl, reduce_batch=False)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret
