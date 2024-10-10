import os
import pathlib
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from utils import printlog, DATASETS_INFO, check_module_prefix, is_distributed, get_rank
from .openai import pretrained_configs, MODEL_URLS, download_pretrained_from_url
from .modified_resnet import ModifiedResNet
from .openclip_vit import VisionTransformer
from models.vit.ViT_layers import trunc_normal_


def make_conv_or_linear(layer, init_weight=None, init_bias=None):
    if init_weight is not None:
        init_weight(tensor=layer.weight.data)
    if init_bias is not None:
        init_bias(tensor=layer.bias.data)
    return layer


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    input_patchnorm: bool = False  # whether to use dual patchnorm - would only apply the input layernorm on each patch, as post-layernorm already exist in original clip vit design
    global_average_pool: bool = False  # whether to global average pool the last embedding layer, instead of using CLS token (https://arxiv.org/abs/2205.01580)
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False

    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


def _build_vision_tower(name, img_size=224, pool_type='tok'):
    cfg = pretrained_configs[name]
    vision_cfg = CLIPVisionCfg(**cfg['vision_cfg'])
    printlog(f"Building vision tower for {name} with cfg {cfg}")
    # local debug
    # vision_cfg.image_size = 64

    # todo add vit
    if 'RN' in name:
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(layers=vision_cfg.layers,
                                output_dim=cfg['embed_dim'],
                                heads=vision_heads,
                                image_size=vision_cfg.image_size,
                                width=vision_cfg.width
                                )
    else:
        #         {
        #             "patch_size": 16,
        #             "embed_dim": 768,
        #             "depth": 12,
        #             "num_heads": 12
        #         },

        visual = VisionTransformer(image_size=img_size,
                                   patch_size=16,
                                   width=768,
                                   layers=12,
                                   heads=12,
                                   mlp_ratio=4,
                                   pool_type=pool_type,
                                   attentional_pool=False,
                                   pos_embed_type='learnable'
                                   )
    return visual


class CLIP(nn.Module):
    """
    CLIP-like model with two visual modalities
    """
    output_dict: torch.jit.Final[bool]

    def __init__(self, config, experiment=1):
        super().__init__()
        self.config = config
        self.dataset = config['dataset']
        self.task = config.get('task', 'detection')  # todo remove as this has no effect
        self.backbone_name = config.get('backbone')
        self.phase = config.get('phase', 'pretraining')
        self.out_stride = config.get('out_stride', 16)
        self.norm = config.get('norm', None)
        self.num_classes = DATASETS_INFO[self.dataset].NUM_CLASSES[experiment]
        self.modalities = config.get("modalities")
        self.modality_mapping = {}
        if self.phase == 'pretraining':
            assert len(self.modalities) == 2, "CLIP model requires two modalities for phase: {self.phase}"

        letters = ['a', 'b']
        if self.config.get('pretrain_modalities', None) is None:
            pretrain_modalities = self.modalities
        else:
            pretrain_modalities = self.config.get('pretrain_modalities', self.modalities)
        for let, modality in zip(letters, pretrain_modalities):
            self.modality_mapping[modality] = let

        self.output_dict = config.get("output_dict", False)
        self.cast_dtype = config.get("cast_dtype", None)
        self.embed_dim = config.get("embed_dim", 256)
        self.quick_gelu = config.get("quick_gelu", False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # log(1/temperature)

        self.backbone_settings = config.get("backbone_settings", {})
        self.backbone_settings['img_size'] = self.backbone_settings.get('img_size', 224)
        printlog(f"CLIP specified backbone settings: {self.backbone_settings}")
        ########################################## checkpoint loading ##################################################
        self.backbone_pretrained = config.get('pretrained', False)
        # determine path to pretrained backbone checkpoint if used
        internal_checkpoint_name = self.config.get("internal_checkpoint_name", None)
        internal_checkpoint_dir = self.config.get("internal_checkpoint_dir", None)

        self.external_backbone_pretrained = False
        self.internal_backbone_pretrained = internal_checkpoint_dir is not None and internal_checkpoint_name is not None
        self.internal_checkpoint_path = None

        if self.internal_backbone_pretrained and self.backbone_pretrained:
            self.internal_checkpoint_path = os.path.join(internal_checkpoint_dir, internal_checkpoint_name)
            printlog(f"Given graph.pretrained {self.backbone_pretrained} and determined internal_backbone_pretrained:"
                     f" {self.internal_backbone_pretrained} --> loading from {self.internal_checkpoint_path}")
        elif self.backbone_pretrained:
            printlog(f"Given graph.pretrained {self.backbone_pretrained} and determined internal_backbone_pretrained:"
                     f" {self.internal_backbone_pretrained} --> downloading init from openai")
            self.external_backbone_pretrained = True
        else:
            printlog(f"Given graph.pretrained {self.backbone_pretrained} and determined internal_backbone_pretrained:"
                     f" {self.internal_backbone_pretrained} and external_backbone_pretrained:"
                     f" {self.external_backbone_pretrained} --> training from scratch")

        self.train_head_only = self.phase in ['linear_probing']
        self._get_backbones()

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

    def _get_backbones(self):
        modalities_in_checkpoint = self.config.get('pretrain_modalities', [self.modalities[0]])

        if self.task not in ['detection', 'classification']:
            raise ValueError(f"task: {self.task} not supported yet")
        if self.phase == 'pretraining':
            # todo pretraining always defaults to a img_size = 224
            #  need to implement interpolation of positional embeddings for different img_size
            #  from openai checkpoint learnable pos_embed
            self.visual_a = _build_vision_tower(self.backbone_name)
            self.visual_b = _build_vision_tower(self.backbone_name)
        elif self.phase in ['finetuning', 'linear_probing']:
            # single modality finetuning
            if len(self.modalities) == 1:
                # by convention OCT is visual_a, IR is visual_b todo generalize this for future pretraining
                if 'OCT' == self.modalities[0]:
                    # fixme : assume vit_base only for now
                    pool_type = self.config['backbone_settings'].get('classifier_feature', 'avg')
                    self.visual_a = _build_vision_tower(self.backbone_name,
                                                        img_size=self.backbone_settings['img_size'],
                                                        pool_type=pool_type)
                    self.visual_b = None
                    self.embed_dim = self.visual_a.output_dim
                    # fixme override for test
                    self.embed_dim = 2*768 if pool_type == 'mean_max_pool' else 768
                    # fixme : assume vit_base only for now
                    self.visual_a.proj = None
                elif 'IR' == self.modalities[0]:
                    self.visual_a = None
                    self.visual_b = _build_vision_tower(self.backbone_name, img_size=self.backbone_settings['img_size'])
                    self.embed_dim = self.visual_b.output_dim
                printlog(f"single-modality {self.modalities} of checkpoint pretrained with: {modalities_in_checkpoint}")

            elif len(self.modalities) == 2:

                # todo implement multi-modality finetuning with other fusion strategies
                #  for now we just concatenate the embeddings
                self.visual_a = _build_vision_tower(self.backbone_name, img_size=self.backbone_settings['img_size'])
                self.visual_b = _build_vision_tower(self.backbone_name, img_size=self.backbone_settings['img_size'])
                self.embed_dim = self.visual_a.output_dim + self.visual_b.output_dim

        else:
            raise NotImplementedError(f"{self.phase} not yet implemented")

        self.head = make_conv_or_linear(layer=torch.nn.Linear(in_features=self.embed_dim, out_features=self.num_classes),
                                        init_bias=partial(torch.nn.init.zeros_),
                                        init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05)
                                        )

        if self.internal_checkpoint_path:
            self._load_internal_pretrained_backbone()

        elif self.external_checkpoint_path:
            self._load_external_pretrained_backbone()

        if self.phase == 'linear_probing':
            for m in self.modalities:
                self.lock_image_tower(modality=m, unlocked_groups=0, freeze_bn_stats=True)

    def lock_image_tower(self, modality, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        modality = self.modality_mapping[modality]
        getattr(self, f'visual_{modality}').lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)
        printlog(f"locked image encoder for modality: {modality}")
        a = 1

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)

    def encode_image(self, image, modality, normalize: bool = False):
        features = getattr(self, f'visual_{modality}')(image)
        return F.normalize(features, dim=-1) if normalize else features

    def _load_external_pretrained_backbone(self):
        model_path = download_pretrained_from_url(MODEL_URLS[self.backbone_name])
        checkpoint_state_dict = torch.jit.load(model_path, map_location='cpu').visual.state_dict()
        missing_keys_from_chkpt_a, missing_keys_from_model_a = self.visual_a.load_state_dict(checkpoint_state_dict,
                                                                                             strict=False)
        missing_keys_from_chkpt_b, missing_keys_from_model_b = self.visual_b.load_state_dict(checkpoint_state_dict,
                                                                                             strict=False)

        # printlog(f'{len(missing_keys_from_chkpt)} keys in model but NOT in chkpt: {missing_keys_from_chkpt}')
        # printlog(f'{len(missing_keys_from_model)} keys in chkpt but NOT in model: {missing_keys_from_model}')
        printlog(f"************************************")
        printlog(f"loading checkpoint for {self.backbone_name} from openai:")
        printlog(f'{len(missing_keys_from_chkpt_a)} keys in model a but NOT in chkpt: {missing_keys_from_chkpt_a}')
        printlog(f'{len(missing_keys_from_model_a)} keys in chkpt but NOT in model: {missing_keys_from_model_a}')
        printlog(f"="*10)
        printlog(f'{len(missing_keys_from_chkpt_b)} keys in model a but NOT in chkpt: {missing_keys_from_chkpt_b}')
        printlog(f'{len(missing_keys_from_model_b)} keys in chkpt but NOT in model: {missing_keys_from_model_b}')
        printlog(f"************************************")

    def _get_pos_embeddings_internal(self, checkpoint_state_dict):
        # todo assume now that num_patches and patches_layout is the same for both modalities
        # pe in checkpoint named visual_a.positional_embedding (1+L,D) i.e includes embedding for CLS token
        pos_embed_checkpoint = checkpoint_state_dict[f"visual_{self.modality_mapping[self.modalities[0]]}.positional_embedding"]
        pos_embed_current = getattr(self, f"visual_{self.modality_mapping[self.modalities[0]]}").positional_embedding
        return pos_embed_current, pos_embed_checkpoint

    def _get_pos_embed_info(self, img_size):
        """returns interpolation function and num_patches_target (i.e current num_patches) """
        # todo assume now that num_patches and patches_layout is the same for both modalities
        printlog(f"Input crop size : {img_size} to infer num_patches_target")
        kernel_size = getattr(self, f"visual_{self.modality_mapping[self.modalities[0]]}").conv1.kernel_size
        num_patches_target = img_size//kernel_size[0] * img_size//kernel_size[1]
        interpolator_func = getattr(self, f"visual_{self.modality_mapping[self.modalities[0]]}").interpolate_pos_encoding_2d
        return interpolator_func, num_patches_target

    def _load_internal_pretrained_backbone(self, remove_key_from_checkpoint: Union[str, tuple, None] = None):
        """ loads an internally pretrained backbone )
        :param remove_key_from_checkpoint: when finetuning we typically do not want to load the some chkpt keys
                                           so remove keys that contain  from the checkpoint
        """
        assert self.phase in ['finetuning', 'pretraining', 'linear_probing', 'fpn_probing'],\
            f'init with internal pretrained backbone only supported for phase in [finetuning, pretraining]' \
            f' instead got phase: {self.phase}'
        if remove_key_from_checkpoint is None:
            remove_key_from_checkpoint = ()
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

        ########################################## pos embeddings ######################################################
        pos_embed_current, pos_embed_checkpoint = self._get_pos_embeddings_internal(checkpoint_state_dict)
        interp_func, num_patches_target = self._get_pos_embed_info(self.backbone_settings['img_size'])
        first_patch_idx = 1  # fixme: this is hardcoded for now. openai VisionTransformer always has extra token for CLS
        printlog(f"positional embedding has extra tokens: {first_patch_idx} in current model")
        pos_embed_cls_checkpoint = pos_embed_checkpoint[:first_patch_idx, :]
        # only interpolate the positional embeddings for the patches (not the extra tokens)
        pos_embed_interpolated = interp_func(target_L=num_patches_target,
                                             pos_embed=pos_embed_checkpoint[first_patch_idx:, :].unsqueeze(0))
        pos_embed_interpolated = torch.cat((pos_embed_cls_checkpoint, pos_embed_interpolated.squeeze(0)), dim=0)

        for m in self.modalities:
            checkpoint_state_dict[f'visual_{self.modality_mapping[m]}.positional_embedding'] = pos_embed_interpolated

        if self.phase in ['finetuning', 'linear_probing'] and len(self.modalities) == 1:
            # note: pretrain modality defaults to finetuning modality if not specified
            modalities_in_checkpoint = self.config.get('pretrain_modalities', [self.modalities[0]])
            printlog(f"single-modality {self.modalities} of checkpoint pretrained with: {modalities_in_checkpoint}")
        elif self.phase in ['finetuning', 'linear_probing'] and len(self.modalities) == 2:
            # note: pretrain modalities defaults to finetuning modalities if not specified
            modalities_in_checkpoint = self.config.get('pretrain_modalities', self.modalities)
            printlog(f"multi-modality finetuning of checkpoint with pretrained with: {modalities_in_checkpoint}")
        else:
            raise ValueError(f'phase: {self.phase} and modalities: {self.modalities} not supported')

        # remove modalities that are not in checkpoint
        remove_key_from_checkpoint = list(remove_key_from_checkpoint)
        for modality in modalities_in_checkpoint:
            if modality not in self.modalities:
                remove_key_from_checkpoint.append(f'visual_{self.modality_mapping[modality]}.')
        checkpoint_state_dict = self.remove_keys_from_checkpoint(checkpoint_state_dict,
                                                                 tuple(remove_key_from_checkpoint))
        printlog(f"Loading {checkpoint_name} from {path}")
        missing_keys_from_chkpt, missing_keys_from_model = self.load_state_dict(checkpoint_state_dict, strict=False)
        printlog(f'{len(missing_keys_from_chkpt)} keys in model but NOT in chkpt: {missing_keys_from_chkpt}')
        printlog(f'{len(missing_keys_from_model)} keys in chkpt but NOT in model: {missing_keys_from_model}')
        printlog(f"************************************")

    def forward(self,
                image_a: Optional[torch.Tensor] = None,
                image_b: Optional[torch.Tensor] = None,
                normalize_a: Optional[bool] = True,
                normalize_b: Optional[bool] = True):
        # todo refactor input to be a dict[modality] = image
        if self.phase == 'pretraining':
            ret_dict = self.forward_pretrain(image_a, image_b, normalize_a, normalize_b, return_as_dict=True)
            return ret_dict['image_features_a'], ret_dict['image_features_b'], ret_dict['logit_scale']
            
        elif self.phase in ['finetuning', 'linear_probing'] and (len(self.modalities) == 1):
            ############################################# single modality task #########################################
            image = image_a if image_a is not None else image_b
            normalize = normalize_a if image_a is not None else normalize_b
            logits = self.forward_single_modality(image, self.modalities[0], normalize)
        elif self.phase in ['finetuning', 'linear_probing'] and (len(self.modalities) == 2):
            ############################################# multi modality task ##########################################
            ret_dict = self.forward_pretrain(image_a, image_b, normalize_a, normalize_b, return_as_dict=True)
            # concat embeddings and pass through head
            logits = self.head(torch.cat([ret_dict['image_features_a'], ret_dict['image_features_b']], dim=1))
        else:
            raise ValueError(f"phase: {self.phase} and modalities: {self.modalities} not supported")
        return logits

    def forward_single_modality(self, image, modality, normalize: Optional[bool] = True):
        modality = self.modality_mapping[modality]  # 'a' or 'b
        image_features = self.encode_image(image, modality, normalize=normalize)
        logits = self.head(image_features)
        return logits

    def forward_multi_modality(self, image_a: Optional[torch.Tensor] = None, image_b: Optional[torch.Tensor] = None,
                               normalize_a: Optional[bool] = True, normalize_b: Optional[bool] = True):
        ret_dict = self.forward_pretrain(image_a, image_b, normalize_a, normalize_b, return_as_dict=True)
        logits = self.head(torch.cat([ret_dict['image_features_a'], ret_dict['image_features_b']], dim=1))
        return logits

    def forward_pretrain(self,
                         image_a: Optional[torch.Tensor] = None,
                         image_b: Optional[torch.Tensor] = None,
                         normalize_a: Optional[bool] = True,
                         normalize_b: Optional[bool] = True,
                         return_as_dict: Optional[bool] = False):

        image_features_a = self.encode_image(image_a, 'a', normalize=normalize_a) if image_a is not None else None
        image_features_b = self.encode_image(image_b, 'b', normalize=normalize_b) if image_b is not None else None
        if self.output_dict or return_as_dict:
            return {
                f"image_features_a": image_features_a,
                "image_features_b": image_features_b,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features_a, image_features_b, self.logit_scale.exp()



