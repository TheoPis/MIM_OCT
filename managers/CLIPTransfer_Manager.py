from typing import Union
from abc import ABC
import matplotlib.pyplot as plt
import torch
from torch import nn
from managers.MultiVitSeq_Manager import MultiVitSeqManager
plt.gray()

ParallelType = nn.parallel.DistributedDataParallel
Model = Union[nn.Module, ParallelType]


class CLIPTransferManager(MultiVitSeqManager, ABC):
    @property
    def is_multimodal_downstream(self):
        return len(self.modalities) > 1

    @property
    def modality_mapping(self):
        return {'OCT': 'a', 'IR': 'b'}

    def forward_train_step(self, img, lbl, reduce_batch=True, **kwrargs):
        """
        :param img: [B, C, H, W] tensor
        :param lbl: [B, num_classes] tensor
        :param reduce_batch: whether to reduce the batch dimension in the loss
        :return:
        """
        ret = dict()
        if self.is_multimodal_downstream:
            output = self.model(img[self.modalities[0]], img[self.modalities[1]])
        else:
            if self.modalities[0] == 'OCT':
                output = self.model(img[self.modalities[0]])
            else:
                output = self.model(image_b=img[self.modalities[1]])
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
            output = self.ema_model((img[self.modalities[0]], img[self.modalities[1]]))
        else:
            output = self.model(img[self.modalities[0]], img[self.modalities[1]])

        loss = None
        if not skip_loss:
            loss = self.get_loss(img, output, lbl, reduce_batch=False)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret
