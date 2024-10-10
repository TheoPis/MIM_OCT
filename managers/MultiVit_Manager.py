from abc import ABC
import torch
from managers.Vit_Manager import VitManager
from utils import to_numpy
from typing import Union
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from models import CLIP
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


class MultiVitManager(VitManager, ABC):
    """ Finetuning multimodal models for single modality tasks (e.g. OCT,IR -> OCT)
    Manager for MultiVit models, essentially inherits most things from VitManager
    Differs from VitManager in forward_train_step and forward_val_step by providing modality as input
    """
    @property
    def modality(self):
        assert len(self.config['graph']['modalities']) == 1, 'only one modality is supported with this manager'
        return self.config['graph']['modalities'][0]

    @property
    def sequence_length(self):
        """ Returns the sequence length (i.e number of tokens per example
         of a ViT model using the patch_embed module at its start"""
        # note: we check CLIP first as it is a special case that does not have backbone.encoder etc
        if isinstance(self.flat_model, CLIP):
            # note: we assume all modalities have the same grid_size = (H/h_patch, W/w_patch)
            return self.flat_model.visual_a.grid_size[0] * self.flat_model.visual_a.grid_size[1]
        elif hasattr(self.flat_model.backbone.encoder, 'patch_embed'):
            return self.flat_model.backbone.encoder.patch_embed.num_patches
        elif hasattr(self.flat_model.backbone.encoder, 'patch_embeders'):
            some_modality = list(self.flat_model.backbone.encoder.patch_embeders.keys())[0]
            # note: we assume all modalities have the same patch size
            return self.flat_model.backbone.encoder.patch_embeders[some_modality].num_patches
        raise ValueError(f'Could not find "backbone.encoder.patch_embed" or'
                         f'"backbone.encoder.patch_embeders" or "CLIP.visual_a.grid_size" '
                         f'for model of class:{self.flat_model.__class__.__name__}')

    @property
    def patch_shape(self):

        if isinstance(self.model, ParallelType):
            if hasattr(self.model.module.backbone.trunk, 'patch_embed'):
                return self.model.module.backbone.trunk.patch_embed.patch_size
            elif hasattr(self.model.module.backbone.trunk, 'patch_embeders'):
                some_modality = list(self.model.module.backbone.trunk.patch_embeders.keys())[0]
                return self.model.module.backbone.trunk.patch_embeders[some_modality].patch_size
        else:
            if hasattr(self.model.backbone.trunk, 'patch_embed'):
                return self.model.backbone.trunk.patch_embed.patch_size
            elif hasattr(self.model.backbone.trunk, 'patch_embeders'):
                some_modality = list(self.model.backbone.trunk.patch_embeders.keys())[0]
                # note: we assume all modalities have the same patch size
                return self.model.backbone.trunk.patch_embeders[some_modality].patch_size
        raise ValueError('Could not find patch_embed or patch_embeders in the model')

    def forward_train_step(self, img, lbl, reduce_batch=True, **kwrargs):
        """
        :param img: (B, C, H, W) tensor
        :param lbl: (B, num_classes) tensor
        :param reduce_batch: whether to reduce the batch dimension in the loss
        :return:
        """
        ret = dict()
        with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.use_autocast):
            output = self.model(img, modality=self.modality)
            loss = self.get_loss(img, output, lbl, reduce_batch=reduce_batch)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def forward_val_step(self, img, lbl, skip_loss=False, **kwrargs):
        """
        :param img: (B, C, H, W) tensor
        :param lbl: (B, num_classes) tensor
        :param skip_loss: whether to skip the loss calculation
        :return:
        """
        ret = dict()
        if self.use_ema and self.ema_model is not None:
            output = self.ema_model(img, modality=self.modality)
        else:
            output = self.model(img, modality=self.modality)
        loss = None
        if not skip_loss:
            loss = self.get_loss(img, output, lbl, reduce_batch=False)
        ret['output'] = output
        ret['loss'] = loss
        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

