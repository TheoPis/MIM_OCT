#!/usr/bin/env python3
import torch
import pathlib
import os
import numpy as np
import torch.nn as nn
from utils import DATASETS_INFO, printlog, is_distributed, get_rank, check_module_prefix
from typing import Union
from timm.models.layers import trunc_normal_
from models.ViT_config import vit_mae_pretraining_single_modality, vit_finetune_single_modality
from models.vit.ViT_layers import AttentionPoolingClassifier


class VitMae(nn.Module):
    eligible_backbones = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']
    eligible_phases = ['pretraining', 'finetuning', 'scratch', 'linear_probing', 'attentive_probing']
    eligible_downstream_phases = ['scratch', 'finetuning', 'linear_probing', 'attentive_probing']

    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.dataset = config['dataset']
        self.task = config.get('task', 'detection')  # todo remove as this has no effect
        self.backbone_name = config.get('backbone')
        self.phase = config.get('phase', 'pretraining')
        self.out_stride = config.get('out_stride', 16)
        self.norm = config.get('norm', None)
        self.num_classes = DATASETS_INFO[self.dataset].NUM_CLASSES[experiment]

        assert(self.backbone_name in self.eligible_backbones), \
            f'backbone_name {self.backbone_name} not in {self.eligible_backbones}'
        assert(self.phase in self.eligible_phases), 'mode must be in {}'.format(self.eligible_phases)

        self.backbone_settings = config.get('backbone_settings', {})
        self.backbone_pretrained = config.get('pretrained', False)

        self.masking_settings = config.get('masking_settings', {})
        self.masking_type = self.masking_settings.get('type', 'random')
        self.masked_ratio = self.masking_settings.get("pred_ratio", 0.75)  # percentage of masked (i.e predicted) pixels
        self.train_head_only = False  # flag used for setting up the optimizer when probing with frozen backbone

        self.modalities = config.get('modalities', ['OCT'])
        # determine path to pretrained backbone checkpoint if used
        internal_checkpoint_name = self.config.get("internal_checkpoint_name", None)
        internal_checkpoint_dir = self.config.get("internal_checkpoint_dir", None)

        external_checkpoint_name = self.config.get("external_checkpoint_name", None)
        external_checkpoint_dir = self.config.get("external_checkpoint_dir", None)

        self.internal_backbone_pretrained = internal_checkpoint_dir is not None and internal_checkpoint_name is not None
        self.internal_checkpoint_path = None

        self.external_backbone_pretrained = external_checkpoint_dir is not None and external_checkpoint_name is not None
        self.external_checkpoint_path = None

        assert not (self.internal_backbone_pretrained and self.external_backbone_pretrained),\
            f'cannot require both internal_backbone_pretrained: ' \
            f'[{self.internal_backbone_pretrained}] ' \
            f'and external_backbone_pretrained:' \
            f' [{self.external_backbone_pretrained}] please select either internal or external in config {self.config}'

        if self.internal_backbone_pretrained and self.backbone_pretrained:
            self.internal_checkpoint_path = os.path.join(internal_checkpoint_dir, internal_checkpoint_name)
            printlog(f"Given graph.pretrained {self.backbone_pretrained} and determined internal_backbone_pretrained:"
                     f" {self.internal_backbone_pretrained} --> loading from {self.internal_checkpoint_path}")

        elif self.external_backbone_pretrained and self.backbone_pretrained:
            self.external_checkpoint_path = os.path.join(external_checkpoint_dir, external_checkpoint_name)
            printlog(f"Given graph.pretrained {self.backbone_pretrained} and determined external_backbone_pretrained:"
                     f" {self.external_backbone_pretrained} --> loading from {self.external_checkpoint_path}")

        if (not self.backbone_pretrained) \
                and (not self.internal_backbone_pretrained) \
                and (not self.external_backbone_pretrained)\
                and self.phase == 'finetuning':
            # if not checkpoint is given and we are in finetuning mode we assume that we are training from scratch
            self.phase = 'scratch'
        self.use_all_seq = False  # needed with multimodal model that uses a single sequence as input
        self._get_backbone()

    def _get_backbone(self):
        if self.backbone_name in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            if self.phase == 'pretraining':
                if self.config.get('cotrain_with_volumes', False):
                    # todo
                    raise NotImplementedError(f'{self.backbone_name} with phase {self.phase} not implemented')
                else:
                    self.backbone = vit_mae_pretraining_single_modality(pretrained=self.backbone_pretrained,
                                                                        backbone=self.backbone_name,
                                                                        **self.config['backbone_settings'])
            elif self.phase == 'finetuning':
                self.backbone = vit_finetune_single_modality(out_features=self.num_classes,
                                                             backbone=self.backbone_name,
                                                             **self.config['backbone_settings'],
                                                             pretrained=self.backbone_pretrained)
            elif self.phase == 'scratch':
                self.backbone = vit_finetune_single_modality(out_features=self.num_classes,
                                                             backbone=self.backbone_name,
                                                             **self.config['backbone_settings'],
                                                             pretrained=False)

            elif self.phase in ['linear_probing', 'attentive_probing']:
                if self.phase == 'attentive_probing':
                    printlog(f"phase = {self.phase} overriding backbone_settings.classifier_feature from "
                             f"{self.config['backbone_settings'].get('classifier_feature', None)} "
                             f"to None to return dense feature maps for attentive pooling")
                    self.config['backbone_settings']['classifier_feature'] = None

                self.backbone = vit_finetune_single_modality(out_features=self.num_classes,
                                                             backbone=self.backbone_name,
                                                             **self.config['backbone_settings'],
                                                             pretrained=self.backbone_pretrained)

        else:
            raise NotImplementedError()
        # todo: does not support spatio-temporal models
        if self.internal_checkpoint_path:
            self._load_internal_pretrained_backbone(remove_key_from_checkpoint=('head', 'heads', 'mask_token', 'decoder'))

        elif self.external_checkpoint_path:
            self._load_external_pretrained_backbone(remove_key_from_checkpoint=None)

        if self.phase == 'linear_probing':
            self.get_linear_probing(self.config.get("use_bn_with_linear_probing", True))
        elif self.phase == 'attentive_probing':
            self.get_attentive_probing()

    def _get_pos_embed_info(self):
        """returns interpolation function and num_patches_target, patches_layout """
        if hasattr(self.backbone, 'encoder'):
            if hasattr(self.backbone.encoder, 'patch_embed'):
                patches_layout = self.backbone.encoder.patch_embed.patches_layout
                num_patches_target = self.backbone.encoder.patch_embed.num_patches
            elif hasattr(self.backbone.encoder, 'patch_embeders'):
                # case of multimodal model with shared encoder and one patch_embeder per modality
                assert isinstance(self.backbone.encoder.patch_embeders, nn.ModuleDict)
                some_modality = list(self.backbone.encoder.patch_embeders.keys())[0]
                # we use some_modality but the following are assumed to be the same for all modalities
                patches_layout = self.backbone.encoder.patch_embeders[some_modality].patches_layout
                num_patches_target = self.backbone.encoder.patch_embeders[some_modality].num_patches
            else:
                raise ValueError(f'backbone [{self.backbone}] has neither patch_embed nor patch_embeders attribute')

            interpolator_func = self.backbone.encoder.interpolate_pos_encoding

        elif hasattr(self.backbone, 'encoders'):
            # case of multimodal model with separate encoders that each contain a patch_embeder per modality
            assert isinstance(self.backbone.encoders, nn.ModuleDict)
            some_modality = list(self.backbone.encoders.keys())[0]
            # we use some_modality but the following are assumed to be the same for all modalities
            patches_layout = self.backbone.encoders[some_modality].patch_embed.patches_layout
            num_patches_target = self.backbone.encoders[some_modality].patch_embed.num_patches
            interpolator_func = self.backbone.encoders[some_modality].interpolate_pos_encoding
        else:
            raise ValueError(f'backbone [{self.backbone}] has neither trunk nor encoders attribute')

        return interpolator_func, num_patches_target, patches_layout

    def _get_pos_embeddings_external(self, checkpoint_state_dict, remove_key_from_checkpoint=None):
        pos_embed_checkpoint = checkpoint_state_dict['pos_embed']
        if hasattr(self.backbone, 'encoders'):
            # we use some_modality but the following are assumed to be the same for all modalities
            some_modality = self.backbone.modalities[0]
            pos_current = self.backbone.encoders[some_modality].pos_embed
        elif hasattr(self.backbone, 'encoder'):
            if self.backbone.encoder.has_temporal_dim:
                # note: we only init from 2D pretrained imagenet checkpoint for now
                #  so we need to skip pos_embed loading
                # if at some point external 3D pretrained checkpoints are available we can remove this hack
                num_temporal_patches = self.backbone.encoder.patch_embed.patches_layout[0]
                num_spatial_patches = np.prod(self.backbone.encoder.patch_embed.patches_layout[1:])
                pe_s = self.backbone.encoder.pos_embed_spatial.repeat(1, num_temporal_patches, 1)
                pe_t = torch.repeat_interleave(self.backbone.encoder.pos_embed_temporal, num_spatial_patches, dim=1)
                # fixme: this is a hack to make it skip loading/interpolating the pos_embed
                # fixme make them equal to skip
                pos_current = pe_s + pe_t
                pos_embed_checkpoint = pos_current
                # also we have to remove patch_embed from checkpoit_state_dict as it is also 2D-specific
                if remove_key_from_checkpoint is None:
                    remove_key_from_checkpoint = []
                remove_key_from_checkpoint.append("patch_embed")
                remove_key_from_checkpoint = tuple(remove_key_from_checkpoint)

                checkpoint_state_dict['pos_embed'] = pos_embed_checkpoint

            else:
                pos_current = self.backbone.encoder.pos_embed
                # assume checkpoint_state_dict['pos_embed'] always includes pos_embed_cls at index 0 in MAE and RETFound
                checkpoint_state_dict['pos_embed'] = pos_embed_checkpoint[:, 1:]
                if self.backbone.encoder.use_cls_token:
                    if hasattr(self.backbone.encoder, 'pos_embed_cls'):
                        # split cls token from pos_embed in checkpoint
                        checkpoint_state_dict['pos_embed_cls'] = pos_embed_checkpoint[:, 0].unsqueeze(0)

        else:
            raise NotImplementedError(f"self.backbone should have either encoders or encoder attribute")

        return pos_current, checkpoint_state_dict, remove_key_from_checkpoint

    def _get_pos_embeddings_internal(self, checkpoint_state_dict, remove_key_from_checkpoint=None):

        if 'backbone.encoder.pos_embed' in checkpoint_state_dict:
            # standard case: load the pos_embed (which is typically non-learnable) from the checkpoint
            pos_embed_checkpoint = checkpoint_state_dict['backbone.encoder.pos_embed']
            if self.phase in ['linear_probing', 'finetuning']:
                pos_embed_cls_checkpoint = checkpoint_state_dict.get('backbone.encoder.pos_embed_cls', None)
                if pos_embed_cls_checkpoint is not None:
                    printlog("Warning: found a legacy internal checkpoint with pos_embed_cls.  this is no longer used")
            else:
                pos_embed_cls_checkpoint = None

            pos_embed_current = self.backbone.encoder.pos_embed

        elif self.phase == 'finetuning' and 'backbone.encoder.pos_embed' in self.state_dict():
            # case where checkpoint is from a OCT-volumetric (3D patch) model and finetuning is done on 2D OCT slices
            # as a result the pos_embed is now just 2D whereas the pretraining pos_embed was 2D (spatial) + 1D (time)
            # we set them the same to avoid any other changes to pos_embed

            # fixme: this is a hack to have backward compatibility with old checkpoints:
            #  the old checkpoints, when use_cls_token=True had a pos_embed of shape (1, 1+L, 768)
            #  where L is the number of patches + 1 (for the cls token)
            #  whereas now the pos_embed is of shape (1, L, 768) and
            #  the pos_embed_cls is added separately on the fly

            _, _, D = checkpoint_state_dict['backbone.encoder.pos_embed_spatial'].shape
            pos_embed_checkpoint = torch.cat([
                # if use_cls_token during pretraining append else empty
                checkpoint_state_dict.get('backbone.encoder.pos_embed_cls', torch.zeros(1, 0, D)),
                checkpoint_state_dict['backbone.encoder.pos_embed_spatial']],
                dim=1)
            pos_embed_current = self.backbone.encoder.pos_embed
            from_3D_to_2D_pos_embed = True

        else:
            # case where model has multiple "encoders" attribute so pos_embed is taken from any of the encoders
            # this is expected when the checkpoint is from a multimodal model with separate modality-specific encoders
            # but finetuning is done on just one of them
            some_modality = list(self.backbone.encoders.keys())[0]
            pos_embed_checkpoint = checkpoint_state_dict[f'backbone.encoders.{some_modality}.pos_embed']
            pos_embed_current = self.backbone.encoders[some_modality].pos_embed

        checkpoint_state_dict['backbone.encoder.pos_embed'] = pos_embed_checkpoint
        return pos_embed_current, checkpoint_state_dict

    def _load_external_pretrained_backbone(self, remove_key_from_checkpoint: Union[str, tuple, None] = None):
        """ loads an externally pretrained backbone (e.x from ssl on ImageNet)
        :param remove_key_from_checkpoint: when finetuning we typically do not want to load the some chkpt keys
                                           so remove keys that contain  from the checkpoint
        """
        printlog(f"*** Loading external pretrained backbone ***")
        assert self.phase in [ph for ph in self.eligible_downstream_phases if ph != 'scratch'],\
            f"init with external pretrained backbone only supported for phases in " \
            f"{[ph for ph in self.eligible_downstream_phases if ph != 'scratch']}" \
            f" instead got phase: {self.phase}"

        if 'vit' not in self.external_checkpoint_path and 'RETFound' not in self.external_checkpoint_path:
            raise NotImplementedError(f'external checkpoint {self.external_checkpoint_path} not supported')

        assert self.config['backbone_settings']['in_channels'] == 3, f'external checkpoint was trained on rgb images,'\
                                                                     f' please specify' \
                                                                     f' graph.backbone_settings.in_channels=3 in config'
        assert self.config['backbone_settings']['use_cls_token'], f'external checkpoint was trained with use_cls_token'\
                                                                  f' please specify' \
                                                                  f' graph.backbone_settings.use_cls_token=true ' \
                                                                  f'in config'

        checkpoint_state_dict = torch.load(self.external_checkpoint_path, map_location='cpu')['model']
        if 'RETFound' in self.external_checkpoint_path:
            checkpoint_state_dict = self.remove_keys_from_checkpoint(checkpoint_state_dict, 'decoder')
        # get pos_embed from current model and pos_embed from checkpoint
        pos_embed_current, checkpoint_state_dict, remove_key_from_checkpoint = \
            self._get_pos_embeddings_external(checkpoint_state_dict, remove_key_from_checkpoint)
        interpolator_func, num_patches_target, patches_layout = self._get_pos_embed_info()

        if pos_embed_current.shape[1] != checkpoint_state_dict['pos_embed'].shape[1]:
            h_chkpt = np.sqrt(checkpoint_state_dict['pos_embed'].shape[1])
            printlog(f"Interpolating PE from {checkpoint_state_dict['pos_embed'].shape[1]} {(1, h_chkpt, h_chkpt)} to "
                     f"{pos_embed_current.shape[1]} {patches_layout}")

            # interpolation
            pos_embed_checkpoint_interpolated = interpolator_func(
                num_patches_target,
                checkpoint_state_dict['pos_embed'],
                patches_layout,
                first_patch_idx=self.backbone.encoder.first_patch_idx)

            checkpoint_state_dict['pos_embed'] = pos_embed_checkpoint_interpolated

        # if single encoder and multiple patch_embedders then init each patch_embedder with the same checkpoint

        if hasattr(self.backbone, 'encoder'):
            if hasattr(self.backbone.encoder, 'patch_embeders'):
                for m in self.backbone.modalities:
                    checkpoint_state_dict[f'patch_embeders.{m}.proj.weight'] = checkpoint_state_dict['patch_embed.proj.weight']
                    checkpoint_state_dict[f'patch_embeders.{m}.proj.bias'] = checkpoint_state_dict['patch_embed.proj.bias']
                if remove_key_from_checkpoint is None:
                    remove_key_from_checkpoint = ('patch_embed.proj')
                elif type(remove_key_from_checkpoint) == tuple:
                    remove_key_from_checkpoint += ('patch_embed.proj')
        if remove_key_from_checkpoint is not None:
            checkpoint_state_dict = self.remove_keys_from_checkpoint(checkpoint_state_dict,
                                                                     remove_key_from_checkpoint)

        printlog(f"Loading from {self.external_checkpoint_path}")
        if hasattr(self.backbone, 'encoders'):
            # case of multimodal model with separate encoders,
            # with each modality encoder initialized with the same external checkpoint
            for modality in self.backbone.modalities:
                missing_keys_from_chkpt, missing_keys_from_model =\
                    self.backbone.encoders[modality].load_state_dict(checkpoint_state_dict, strict=False)
                printlog(f"Init of {modality} encoder with external checkpoint")
                printlog(f'{len(missing_keys_from_chkpt)} keys in model but NOT in chkpt: {missing_keys_from_chkpt}')
                printlog(f'{len(missing_keys_from_model)} keys in chkpt but NOT in model: {missing_keys_from_model}')
                printlog(f"-"*10)
        elif hasattr(self.backbone, 'encoder'):
            # case of unimodal or multimodal model with shared encoder
            missing_keys_from_chkpt, missing_keys_from_model = \
                self.backbone.encoder.load_state_dict(checkpoint_state_dict, strict=False)
            printlog(f'{len(missing_keys_from_chkpt)} keys in model but NOT in chkpt: {missing_keys_from_chkpt}')
            printlog(f'{len(missing_keys_from_model)} keys in chkpt but NOT in model: {missing_keys_from_model}')
            printlog(f"************************************")
        else:
            raise ValueError(f'backbone [{self.backbone}] has neither "encoder" nor "encoders" attribute')

    def _load_internal_pretrained_backbone(self, remove_key_from_checkpoint: Union[str, tuple, None] = None):
        """ loads an internally pretrained backbone (e.x from ssl on KEKI)
        :param remove_key_from_checkpoint: when finetuning we typically do not want to load the some chkpt keys
                                           so remove keys that contain  from the checkpoint
        """
        assert self.phase in [ph for ph in self.eligible_downstream_phases if ph != 'scratch'],\
            f"init with external pretrained backbone only supported for phases in " \
            f"{[ph for ph in self.eligible_downstream_phases if ph != 'scratch']}" \
            f" instead got phase: {self.phase}"

        printlog(f"*** Loading internal pretrained backbone ***")
        internal_checkpoint_path = pathlib.Path(self.internal_checkpoint_path)
        checkpoint_list = [f.name for f in (internal_checkpoint_path / 'chkpts').iterdir()]
        checkpoint_list.sort()
        chkpt_type = 'last'
        printlog(f"checkpoint dir : \n{checkpoint_list}")
        # for backwards compatibility
        if 'chkpt_epoch_best.pt' in checkpoint_list:
            i = checkpoint_list.index('chkpt_epoch_best.pt')
            checkpoint_list.pop(i)
        # current naming convention for best checkpoint
        if 'chkpt_best.pt' in checkpoint_list:
            i = checkpoint_list.index('chkpt_best.pt')
            checkpoint_list.pop(i)

        assert (len(checkpoint_list) > 0)
        if 'chkpt_epoch_' in checkpoint_list[-1]:
            checkpoint_name = checkpoint_list[-1]
        else:
            raise ValueError("No checkpoint of type 'last' found.")
        path = internal_checkpoint_path / 'chkpts' / checkpoint_name
        map_location = {'cuda:%d' % 0: 'cuda:%d' % get_rank()} if is_distributed() else None
        checkpoint_state_dict = check_module_prefix(model_state_dict=self.state_dict(),
                                                    chkpt_state_dict=torch.load(path, map_location=map_location)
                                                    ['model_state_dict'])

        from_3D_to_2D_pos_embed = False
        pos_embed_cls_checkpoint = None

        # get pos_embed from current model and pos_embed from checkpoint
        pos_embed_current, checkpoint_state_dict = self._get_pos_embeddings_internal(checkpoint_state_dict)
        interpolator_func, num_patches_target, patches_layout = self._get_pos_embed_info()

        if pos_embed_current.shape[1] != checkpoint_state_dict['backbone.encoder.pos_embed'].shape[1]:
            h_chkpt = np.sqrt(checkpoint_state_dict['backbone.encoder.pos_embed'].shape[1])

            printlog(f"Interpolating PE from {checkpoint_state_dict['backbone.encoder.pos_embed'].shape[1]}"
                     f" {(1, h_chkpt, h_chkpt)} to {pos_embed_current.shape[1]} {patches_layout}")

            pos_embed_checkpoint = interpolator_func(
                num_patches_target,
                checkpoint_state_dict['backbone.encoder.pos_embed'],
                patches_layout)

            if from_3D_to_2D_pos_embed:
                # we need to remove the cls token pos embed from the checkpoint
                # because it is added separately in more recent checkpoints
                pos_embed_checkpoint = pos_embed_checkpoint[:, 1:, :]

            if hasattr(self.backbone, 'encoders'):
                for modality in self.backbone.encoders.keys():
                    checkpoint_state_dict[f'backbone.encoders.{modality}.pos_embed'] = pos_embed_checkpoint
            elif hasattr(self.backbone, 'encoder'):
                checkpoint_state_dict['backbone.encoder.pos_embed'] = pos_embed_checkpoint
                # checkpoint_state_dict['backbone.encoder.pos_embed_cls'] = pos_embed_checkpoint[:, :first_patch_idx, :]
            else:
                raise ValueError(f'backbone does not have trunk or encoders attribute')

        if remove_key_from_checkpoint is not None:
            checkpoint_has_separate_encoders = False
            for key in checkpoint_state_dict.keys():
                if 'backbone.encoders' in key:
                    checkpoint_has_separate_encoders = True
                    break

            # if self.phase == 'finetuning' and len(self.modalities) == 1 and checkpoint_has_separate_encoders:
            if self.phase == 'finetuning' and len(self.modalities) == 1 and checkpoint_has_separate_encoders:
                printlog(f"single-modality {self.modalities} finetuning of a multimodal checkpoint:"
                         " removing unused modality-specific keys from checkpoint")
                # single-modality finetuning of a multimodal model
                # find modalities in checkpoint from checkpoint keys
                # then remove modality-specific keys from checkpoint for modalities not in self.modalities (finetuning)
                modalities_in_checkpoint = []
                for key in checkpoint_state_dict.keys():
                    if 'backbone.encoders' in key:
                        # keys of the form: "backbone.encoders.modality.*"
                        m = key.split('.')[2]
                        # if m not in ['{}']:
                        modalities_in_checkpoint.append(m)

                modalities_in_checkpoint = list(set(modalities_in_checkpoint))

            elif self.phase in ['finetuning', 'linear_probing', 'attentive_probing'] and len(self.modalities) == 1:
                # note: pretrain modality defaults to finetuning modality if not specified
                modalities_in_checkpoint = self.config.get('pretrain_modalities', [self.modalities[0]])
                printlog(f"single-modality {self.modalities} of checkpoint pretrained with: {modalities_in_checkpoint}")
            elif self.phase in ['finetuning', 'linear_probing', 'attentive_probing'] and len(self.modalities) == 2:
                # note: pretrain modalities defaults to finetuning modalities if not specified
                modalities_in_checkpoint = self.config.get('pretrain_modalities', self.modalities)
                printlog(f"multi-modality finetuning of checkpoint with pretrained with: {modalities_in_checkpoint}")
            else:
                raise ValueError(f'phase: {self.phase} and modalities: {self.modalities} not supported')

            # remove modalities that are not in checkpoint
            remove_key_from_checkpoint = list(remove_key_from_checkpoint)
            # fixme: remove legacy unused keys from checkpoint: pos_embed_cls and pos_embed_{modality}
            remove_key_from_checkpoint.append('pos_embed_cls')
            for modality in modalities_in_checkpoint:
                # fixme remove pos_embed_{modality} from checkpoint
                remove_key_from_checkpoint.append(f'pos_embed_{modality}')
                if modality not in self.modalities:
                    remove_key_from_checkpoint.append(modality)

            checkpoint_state_dict = self.remove_keys_from_checkpoint(checkpoint_state_dict,
                                                                     tuple(remove_key_from_checkpoint))
        printlog(f"Loading {checkpoint_name} from {path}")
        missing_keys_from_chkpt, missing_keys_from_model = self.load_state_dict(checkpoint_state_dict, strict=False)
        printlog(f'{len(missing_keys_from_chkpt)} keys in model but NOT in chkpt: {missing_keys_from_chkpt}')
        printlog(f'{len(missing_keys_from_model)} keys in chkpt but NOT in model: {missing_keys_from_model}')
        printlog(f"************************************")

    @staticmethod
    def remove_keys_from_checkpoint(checkpoint_state_dict, remove_keywords: Union[str, tuple]):
        if type(remove_keywords) == str:
            remove_keywords = (remove_keywords,)
        for keyword in remove_keywords:
            printlog(f"removing keys that contain '{keyword}' ...")
            printlog(f"initial checkpoint state_dict has {len(checkpoint_state_dict)} keys")
            # remove keys in state dict that contain keyword
            checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if keyword not in k}
            printlog(f"new checkpoint state_dict has {len(checkpoint_state_dict)} keys")
        return checkpoint_state_dict

    def get_linear_probing(self, use_bn):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal """
        # freeze all but the head
        for _, p in self.backbone.encoder.named_parameters():
            p.requires_grad = False
        for _, p in self.backbone.head.named_parameters():
            p.requires_grad = True
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Going to use linear probing --> trainable parameters: {n_trainable_parameters} "
                 f"-- frozen parameters: {n_frozen_parameters}")

        # re-init head
        printlog(f"re-init head with trunc_normal")
        trunc_normal_(self.backbone.head.weight, std=0.01)
        self.train_head_only = True
        if use_bn:
            self.backbone.head = nn.Sequential(
                torch.nn.Sequential(torch.nn.BatchNorm1d(self.backbone.head.in_features,
                                                         affine=False,
                                                         eps=1e-6),
                                    self.backbone.head
                                    )
            )
            printlog("Using BN in linear probing head")

    def get_attentive_probing(self):
        """ Sets the backbone to frozen adds AttentionPoolingClassifier  """
        # todo unhardcode this:
        if self.backbone.head.in_features == 1024:
            num_heads = 16  # to ensure divisibility of dim by num_heads
        elif self.backbone.head.in_features == 768:
            num_heads = 12
        else:
            raise NotImplementedError(f"backbone head.in_features {self.backbone.head.in_features} not supported")

        self.backbone.head = AttentionPoolingClassifier(self.backbone.head.in_features,
                                                        self.num_classes,
                                                        num_heads=num_heads)
        # freeze all but the head
        for _, p in self.backbone.encoder.named_parameters():
            p.requires_grad = False
        for _, p in self.backbone.head.named_parameters():
            p.requires_grad = True
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Going to use attentive probing --> trainable parameters: {n_trainable_parameters} "
                 f"-- frozen parameters: {n_frozen_parameters}")

        self.train_head_only = True

    @staticmethod
    def generate_mask(batch_size, sequence_length, masked_ratio, device):
        """ generate an independent random mask for each sequence (i.e for each tokenized image) in the batch
        :param batch_size: batch size (B)
        :param sequence_length: sequence length (L)
        :param masked_ratio: percentage of masked tokens
        :param device: torch.device
        :return: mask(B, L) and ids_restore: (B, L)
        """
        num_keep = int(sequence_length * (1 - masked_ratio))
        noise = torch.rand(batch_size, sequence_length, device=device)  # (B, L)
        ids_shuffle = torch.argsort(noise, dim=1)  # sort ascending: high values are removed
        # note: ids_restore contains the permutation (per batch element) that if applied to the sequence
        # concat(tokens, masked_tokens) would restore its original order
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # ids_keep = ids_restore[:, :num_keep]  # (B, num_keep) remove last L-num_keep values  # todo remove unused
        mask = torch.ones([batch_size, sequence_length], device=device)  # 0 keep , 1 remove
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)  # (B, num_keep)
        mask_dict = {'mask': mask, 'ids_restore': ids_restore}
        return mask_dict

    def get_dims(self, x):
        """ determine batch size, sequence length and device from a tensor x or a dict of tensors x """
        L, B, device = 0, 0, None  # to supress complaints
        if self.use_all_seq and isinstance(x, dict):
            for modality in x:
                x_m = x[modality]
                B, L_m = self._get_dims(x_m)
                L += L_m
                device = x_m.device
        elif not self.use_all_seq and isinstance(x, torch.Tensor):
            B, L = self._get_dims(x)
            device = x.device
        else:
            raise NotImplementedError(f"invalid combination of x type {type(x)} and use_all_seq {self.use_all_seq}")
        return B, L, device

    def _get_dims(self, x):
        """determine batch size and sequence length from a tensor x"""
        dims = x.shape
        if len(dims) == 5:
            B, C, T, H, W = dims
            L = T // self.backbone.patch_size_temporal * H // self.backbone.patch_size * W // self.backbone.patch_size
        elif len(dims) == 4:
            B, C, H, W = dims
            L = H // self.backbone.patch_size * W // self.backbone.patch_size  # sequence length
        else:
            raise NotImplementedError("only 2D (B C H W) and 3D (B C T H W) images are supported")
        return B, L

    def forward(self, x, is_training=True, debug=False, **kwargs):
        if is_training and self.phase == "pretraining":
            B, L, device = self.get_dims(x)
            mask_dict = self.generate_mask(B, L, self.masked_ratio, device)
            out = self.backbone(x, mask_dict)
            if debug:
                return out, mask_dict
            return out, mask_dict['mask']

        else:
            out = self.backbone(x)
        return out


class VitMaeVol(VitMae):

    def __init__(self, config, experiment):
        super().__init__(config, experiment)
        self.backbone.head = nn.Identity()
        # self.volume_fusion_head = ...
        # todo STAGE : this depends on the task, passed as config['task']
        # e.g. if task is a separate regression per location
        num_locations = 10
        # prediction head gives you 10 different values
        self.prediction_head = nn.Linear(self.backbone.encoder.embed_dim, num_locations)
        # there could also be an activation function on top that limits the range of values
        # (e.g a sigmoid for [0,1], tanh for [-1, 1] etc)

        # todo STAGE : to think about
        # if task is classification per location things are more complicated ...
        # target output would have to be (B, num_locations, num_classes_per_location)
        # with num_classes_per_location being {min_integer_value, ... , max_integer_value}
        # this cannot be simply obtained using a linear layer.
        # one idea to still make use a linear layer would be
        # to flatten the target output to a vector of shape (num_locations*num_classes_per_location)
        # and then in the loss unflatten this to compute a cross-entropy loss per location

    def forward(self, x, is_training=True, debug=False, **kwargs):
        # x (B, C, T, H, W) # B batch size , C channels = 3 , T number of Bscans, H height, W width
        B, C, T, H, W = x.shape
        out = []
        # out = torch.zeros(B, self.num_classes, T).cuda()
        for slice_id in range(T):
            out.append(self.backbone(x[:, :, slice_id, :, :]))
        out = torch.stack(out, dim=2)
        # fuse across each volume
        out = out.mean(dim=2)
        # prediction head
        logits = self.prediction_head(out)
        return logits
