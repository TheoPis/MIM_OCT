#!/usr/bin/env python3
from typing import Union, Dict
import torch
from torch import nn
from functools import partial
from utils import printlog
from timm.models.layers import trunc_normal_
from models.ViT_config import\
    vit_mae_pretraining_multi_modality,\
    mvit_finetune_single_modality, \
    mvit_finetune_multi_modality
from models.mae.MAE import VitMae
from models.vit.ViT_layers import AttentionPoolingClassifier, AttentionPoolingClassifierMultimodal


class MultiVitMae(VitMae):
    """ Multimodal extension of VitMae :
    - encoder is shared between modalities except patch_embeder (tokenization is done per modality)
    - decoder is either separate for each modality or shared with separate heads
    - supports multimodal pretraining with two input options:
        - use_all_seq = True, each input is a sequence of tokens from all modalities
        - use_all_seq = False, each input is a sequence of tokens from a single modality and alternates
    - extends linear probing and attentive pooling to multimodal cases
    """
    def __init__(self, config, experiment):
        self.train_modality_tokens = False  # whether to train the modality tokens when using multimodal linear probing
        super().__init__(config, experiment)  # calls self._get_backbone
        self.setup_masking_settings()

    def setup_masking_settings(self):
        """ Setup masking settings for pretraining phase """

        if self.phase == 'pretraining':
            if isinstance(self.masked_ratio, list):
                if len(self.masked_ratio) != len(self.modalities):
                    raise ValueError(f'Masked Ratio: '
                                     f'len(masked_ratio)={len(self.masked_ratio)}!=len(modalities)={len(self.modalities)}')
                if not all(i == self.masked_ratio[0] for i in self.masked_ratio):
                    masked_ratio_dict = dict(zip(self.modalities, self.masked_ratio))
                    self.masked_ratio = masked_ratio_dict
                    printlog(f"Masked ratio varies per modality: {self.masked_ratio}")
                else:
                    masked_ratio_dict = dict(zip(self.modalities, self.masked_ratio))
                    self.masked_ratio = masked_ratio_dict
                    printlog(f"Masked ratio same across modalities : {self.masked_ratio}")

            else:
                printlog(f"Masked ratio applied for the whole sequence: {self.masked_ratio}")
        else:
            printlog(f"Masking not used for phase: {self.phase}")

    def _get_backbone(self):
        self.mm_pretrained = len(self.config.get('pretrain_modalities', [None])) > 1
        printlog(f'Pretrained model is_multimodal: {self.mm_pretrained}')
        if self.mm_pretrained:
            printlog(f'Pretraining modalities were "{self.config["pretrain_modalities"]}"')
            finetune_modalities = self.modalities

        self.use_all_seq = self.config['backbone_settings'].get('use_all_seq', False)
        if self.phase == 'attentive_probing':
            # this forces the backbone ViT to return dense feature maps instead of pooled.
            # pooling happens in the head (AttentionPoolingClassifier)
            printlog(f"phase = {self.phase} overriding backbone_settings.classifier_feature from "
                     f"{self.config['backbone_settings'].get('classifier_feature', None)} "
                     f"to None to return dense feature maps for attentive pooling")
            self.config['backbone_settings']['classifier_feature'] = None

        if self.backbone_name in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            if self.phase == 'pretraining':
                self.backbone = vit_mae_pretraining_multi_modality(pretrained=self.backbone_pretrained,
                                                                   backbone=self.backbone_name,
                                                                   patch_embeders=self.config['patch_embeders'],
                                                                   **self.config['backbone_settings'],
                                                                   modalities=self.modalities)
            elif self.phase in ['finetuning', 'scratch'] and len(self.modalities) == 1:
                self.backbone = mvit_finetune_single_modality(out_features=self.num_classes,
                                                              pretrained=self.backbone_pretrained,
                                                              backbone=self.backbone_name,
                                                              patch_embeders=self.config['patch_embeders'],
                                                              **self.config['backbone_settings'],
                                                              modalities=self.modalities)

            elif self.phase in ['finetuning', 'scratch', 'linear_probing', 'attentive_probing']\
                    and len(self.modalities) == 2:

                self.backbone = mvit_finetune_multi_modality(out_features=self.num_classes,
                                                             pretrained=self.backbone_pretrained,
                                                             backbone=self.backbone_name,
                                                             patch_embeders=self.config['patch_embeders'],
                                                             **self.config['backbone_settings'],
                                                             modalities=self.modalities)

            elif self.phase in ['linear_probing', 'attentive_probing'] and len(self.modalities) == 1:
                self.backbone = mvit_finetune_single_modality(out_features=self.num_classes,
                                                              pretrained=self.backbone_pretrained,
                                                              backbone=self.backbone_name,
                                                              patch_embeders=self.config['patch_embeders'],
                                                              **self.config['backbone_settings'],
                                                              modalities=self.modalities)
            else:
                raise NotImplementedError(f"does not support phase: '{self.phase}' with modalities: '{self.modalities}'"
                                          f" and mm_pretrained: '{self.mm_pretrained}'")
        else:
            raise NotImplementedError(f"backbone_name {self.backbone_name} not supported")

        if self.internal_checkpoint_path:
            self._load_internal_pretrained_backbone(remove_key_from_checkpoint=('head', 'mask_token', 'decoder'))

        elif self.external_checkpoint_path:
            self._load_external_pretrained_backbone(remove_key_from_checkpoint=None)

        if self.phase == 'linear_probing':
            self.get_linear_probing(self.config.get("use_bn_with_linear_probing", True))
        if self.phase == 'attentive_probing':
            self.get_attentive_probing()

    def get_linear_probing(self, use_bn):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal -- assumes single modality finetune"""
        modality_token_names = [f"{modality}_token" for modality in self.modalities]
        if self.external_checkpoint_path is not None and (len(self.modalities) == 2) \
                and self.config['backbone_settings'].get('use_modality_token', False):
            printlog(f"NOT Freezing the modality tokens {modality_token_names} in the encoder for multimodal linear probing")
            self.train_modality_tokens = True

        # freeze all but the head
        for p_name, p in self.backbone.encoder.named_parameters():
            # if using: external_checkpoint, multimodal linear probing, and using a modality token in the encoder
            # skip freezing the modality tokens (this is expected for multimodal linear probing with imagenet)
            if p_name in modality_token_names and self.train_modality_tokens:
                printlog(f" ---> Training {p_name} in the encoder for multimodal linear probing")
                p.requires_grad = True
            else:
                p.requires_grad = False

        for _, p in self.backbone.heads.named_parameters():
            p.requires_grad = True
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Going to use linear probing --> trainable parameters: {n_trainable_parameters} -- frozen parameters: {n_frozen_parameters}")

        # re-init head
        printlog(f"re-init head with trunc_normal")

        if self.phase == 'linear_probing' and len(self.modalities) == 1:
            # single modality finetuning of multimodal-y pretrained model
            trunc_normal_(self.backbone.heads[self.modalities[0]].weight, std=0.01)
            if use_bn:
                self.backbone.heads[self.modalities[0]] = \
                    nn.Sequential(torch.nn.Sequential(
                        torch.nn.BatchNorm1d(self.backbone.heads[self.modalities[0]].in_features, affine=False, eps=1e-6),
                        self.backbone.heads[self.modalities[0]]
                    ),

                )
                printlog("Using BN in linear probing head")

        elif self.phase == 'linear_probing' and len(self.modalities) == 2:
            # multimodal finetuning (many-to-one) of multimodal-y pretrained model
            trunc_normal_(self.backbone.heads.weight, std=0.01)
            if use_bn:
                self.backbone.heads = \
                    nn.Sequential(torch.nn.Sequential(
                        torch.nn.BatchNorm1d(self.backbone.heads.in_features, affine=False, eps=1e-6),
                        self.backbone.heads
                    ),
                    )
                printlog("Using BN in linear probing head")

        self.train_head_only = True

    def get_attentive_probing(self):
        """ Sets the backbone to frozen adds AttentionPoolingClassifier  """

        if len(self.modalities) == 1:
            if self.backbone.heads[self.modalities[0]].in_features == 1024:
                num_heads = 16  # to ensure divisibility of dim by num_heads
            elif self.backbone.heads[self.modalities[0]].in_features == 768:
                num_heads = 12
            else:
                raise NotImplementedError(f"backbone head.in_features {self.backbone.heads[self.modalities[0]].in_features} not supported")

            self.backbone.heads = AttentionPoolingClassifier(self.backbone.heads[self.modalities[0]].in_features,
                                                             self.num_classes,
                                                             num_heads=num_heads)
        else:
            if self.backbone.heads.in_features == 1024:
                num_heads = 16  # to ensure divisibility of dim by num_heads
            elif self.backbone.heads.in_features == 768:
                num_heads = 12
            else:
                raise NotImplementedError(f"backbone head.in_features {self.backbone.head.in_features} not supported")

            self.backbone.heads = AttentionPoolingClassifierMultimodal(self.backbone.heads.in_features,
                                                                       self.num_classes,
                                                                       num_heads=num_heads,
                                                                       modalities=self.modalities,
                                                                       pooling_type=self.config.get('head_pooling',
                                                                                                    'per_modality'))
        self.freeze_backbone()
        self.train_head_only = True

    def freeze_backbone(self):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal """
        # freeze all but the head
        for _, p in self.backbone.encoder.named_parameters():
            p.requires_grad = False
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Freezing backbone --> trainable parameters: {n_trainable_parameters} "
                 f"-- frozen parameters: {n_frozen_parameters}")
        return n_trainable_parameters, n_frozen_parameters

    def generate_mask_per_modality(self, batch_size: int, sequence_length: int, masked_ratio: Dict, device):
        """ Generate an independent random mask for each modality's sequence (i.e. for each tokenized image) with
        different masking ratios per modality. The resulting mask is combined into the same format as generate_mask

        :param batch_size: batch size (B)
        :param sequence_length: sequence length (L)
        :param masked_ratio: percentage of masked tokens for each modality
        :param device: torch.device
        :return: mask(B, L) and ids_restore: (B, L)
        """
        mask_dict = {}
        ids_restore_per_modality = []
        ids_offset = 0
        masks = []
        num_remove = 0
        for modality in self.modalities:
            masked_ratio_m = masked_ratio[modality]
            sequence_length_m = self.backbone.encoder.patch_embeders[modality].num_patches
            num_keep = int(sequence_length_m * (1 - masked_ratio_m))
            num_remove += sequence_length_m - num_keep
            mask, ids_restore = self.generate_mask(batch_size, sequence_length_m, masked_ratio_m, device, True)
            ids_restore_per_modality.append(ids_restore+ids_offset)
            ids_offset += sequence_length_m
            masks.append(mask)
            mask_dict[f'num_keep_{modality}'] = num_keep
        mask_dict.update({'mask': torch.cat(masks, dim=1), 'ids_restore': torch.cat(ids_restore_per_modality, dim=1), })
        if self.use_all_seq:
            _, mask_dict = self._get_masking_info(mask_dict, mask_dict['mask'], sequence_length, num_remove)
        return mask_dict

    def generate_mask(self, batch_size: int, sequence_length: int, masked_ratio: float, device, skip_info=False):
        """ Generate an independent random mask for each sequence (i.e. for each tokenized image or seq of
        tokenized images) in the batch

        :param batch_size: batch size (B)
        :param sequence_length: sequence length (L)
        :param masked_ratio: percentage of masked tokens
        :param device: torch.device
        :param skip_info: if True, skip adding number of masked tokens per modality to mask_dict
        :return: mask(B, L) and ids_restore: (B, L)
        """
        valid_mask_obtained = False
        mask_dict = {}
        while not valid_mask_obtained:
            num_keep = int(sequence_length * (1 - masked_ratio))
            num_remove = sequence_length - num_keep
            noise = torch.rand(batch_size, sequence_length, device=device)  # (B, L)
            ids_shuffle = torch.argsort(noise, dim=1)  # sort ascending: high values are removed
            # note: ids_restore contains the permutation (per batch element) that if applied to the sequence
            # concat(tokens, masked_tokens) would restore its original order
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            # ids_keep = ids_restore[:, :num_keep]  # (B, num_keep) remove last L-num_keep values  # todo remove unused
            mask = torch.ones([batch_size, sequence_length], device=device)  # 0 keep , 1 remove
            mask[:, :num_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)  # (B, L) with correct ordering
            valid_mask_obtained = True
            if self.use_all_seq and not skip_info:
                valid_mask_obtained, mask_dict = self._get_masking_info(mask_dict, mask, sequence_length, num_remove)
            if not valid_mask_obtained:
                printlog(f"invalid mask generated with num_masked_tokens_per_modality:"
                         f" {mask_dict['num_masked_tokens_per_modality']}, retrying ...")

            if not skip_info:
                mask_dict.update({'mask': mask, 'ids_restore': ids_restore})
            else:
                return mask, ids_restore
        return mask_dict

    def _get_masking_info(self, mask_dict, mask, sequence_length: int, num_remove: int):
        """ Get masking info for each modality """
        B = mask.shape[0]
        # note: assume modalities all have the same sequence_length fixme
        sequence_length_m = sequence_length // len(self.modalities)  # L // num_modalities

        # this is for visuals only
        start = 0
        for m in self.modalities:
            end = start + sequence_length_m
            mask_dict[f'mask_{m}'] = mask[:, start:end]
            start = end
        # note: assume modalities all have the same sequence_length fixme
        m_ids = torch.arange(0, len(self.modalities)).unsqueeze(0).T.repeat(B, sequence_length_m).reshape(B, -1)
        m_ids_onehot = torch.nn.functional.one_hot(m_ids.long().to(mask.device), num_classes=len(self.modalities))
        m_sums = (m_ids_onehot * mask.unsqueeze(-1)).sum(dim=1).long()
        mask_dict.update({'num_masked_tokens_per_modality': m_sums})
        # mask_dict.update({'num_kept_tokens_per_modality': sequence_length_per_modality - modality_sums})
        valid_mask_obtained = (m_sums > 0).all().item()  # ensure modality has all tokens masked
        sane_mask_obtained = (m_sums.sum(dim=1) == num_remove).all().item()  # ensure num_remove is correct
        if not sane_mask_obtained:
            raise ValueError(f"something is off with the obtained mask: num_remove is {num_remove} "
                             f"but got modality_sums: {m_sums} with elements summing to {m_sums.sum(dim=1)}")
        return valid_mask_obtained, mask_dict

    def get_masks(self, B, L, device, modality):
        if isinstance(self.masked_ratio, dict) and self.use_all_seq:
            # mask sequence of multiple modalities
            mask_dict = self.generate_mask_per_modality(B, L, self.masked_ratio, device)
        elif isinstance(self.masked_ratio, dict) and not self.use_all_seq:
            # mask sequence of a single modality
            printlog(f"Masking ratio for modality {modality} {self.masked_ratio[modality]}")
            mask_dict = self.generate_mask(B, L, self.masked_ratio[modality], device)
        elif isinstance(self.masked_ratio, float):
            mask_dict = self.generate_mask(B, L, self.masked_ratio, device)
        else:
            raise ValueError(f"masked_ratio must be float or dict, got {type(self.masked_ratio)} "
                             f"with value {self.masked_ratio}")
        return mask_dict

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], is_training=True, **kwargs):
        modality = None
        # if use_all_seq then each input is a seq of all modalities
        if not self.use_all_seq and self.phase not in ['finetuning', 'linear_probing']:
            assert 'modality' in kwargs.keys(), 'modality must be specified in kwargs'
            modality = kwargs['modality']
        outs = {}
        if is_training and self.phase == "pretraining":
            B, L, device = self.get_dims(x)
            mask_dict = self.get_masks(B, L, device, modality)
            out = self.backbone(x, mask_dict, modality)
            # outs['cls_token'] = cls_token
            outs['output'] = out
            outs['mask'] = mask_dict['mask']
            if self.use_all_seq:
                for m in self.modalities:
                    outs[f'mask_{m}'] = mask_dict[f'mask_{m}']
            return outs

        elif isinstance(x, torch.Tensor) and len(self.modalities) == 1:
            # single modality finetuning or inference (OctBiom, DR OCT or IR only) with multimodal-y pretrained model
            # todo remove this hack -> needed by MAE_proto/MutliModalMAE.forward_standard
            if modality is None:
                modality = kwargs.get('modality', self.modalities[0])
                # modality = self.modalities[0]
            out = self.backbone(x, mask=None, modality=modality)
            return out
        elif isinstance(x, dict) and len(self.modalities) > 1:
            # multi-modal (many-to-one) finetuning (x is a Dict[modality,torch.Tensor])
            return_embeddings = kwargs.get('return_embeddings', False)
            if return_embeddings:
                out = self.backbone(x, mask=None, modality=None, return_embeddings=return_embeddings)
                return out
            out = self.backbone(x, mask=None, modality=None)
            return out

        else:
            raise ValueError(f"invalid combination:\n"
                             f" input type: {type(x)}\n"
                             f" modalities: {self.modalities}\n"
                             f" use_all_seq: {self.use_all_seq}\n"
                             f" is_training: {is_training}\n"
                             f" phase: {self.phase}\n")
