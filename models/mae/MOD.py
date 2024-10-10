#!/usr/bin/env python3
from typing import Union, Dict
import torch
from torch import nn
from functools import partial
from utils import printlog
from timm.models.layers import trunc_normal_

from models.ViT_config import \
    vit_mae_pretraining_multi_modality, \
    mvit_finetune_single_modality, \
    mvit_finetune_multi_modality
from models.mae.MAE import VitMae


class MOD(VitMae):
    """ Multimodal version of VitMae that expects training with alternating sgd:
    - embedding is done separately for each modality
    - encoder is shared between modalities except patch_embeder
    - decoder is either separate for each modality (e.g as in omnimae)
              or shared with separate heads (e.g. in use_all_seq mode)
    """

    def __init__(self, config, experiment):
        # this controls whether to train the modality tokens when using multimodal linear probing
        self.train_modality_tokens = False
        super().__init__(config, experiment)
        from torch.distributions.dirichlet import Dirichlet

        # handling multimodal masking settings for pretraining
        if self.phase == 'pretraining':
            if self.masking_type == 'dirichlet':
                assert 'alpha' in self.masking_settings.keys(), f"'alpha' missing in masking_settings dirichlet masking"
                assert 'num_visible_tokens' in self.masking_settings.keys(), \
                    f"'num_visible_tokens' missing in masking_settings dirichlet masking"
                self.alphas = [self.masking_settings['alpha']]*len(self.modalities) \
                    if isinstance(self.masking_settings['alpha'], float) else self.masking_settings['alpha']
                self.num_visible_tokens = self.masking_settings['num_visible_tokens']
                self.distr = Dirichlet(torch.tensor(self.alphas))

                printlog(f"****\nUsing dirichlet masking with settings:"
                         f"\n alpha: {self.alphas}  "
                         f"\n num_visible_tokens {self.num_visible_tokens} "
                         f"\n****")
            else:
                raise NotImplementedError(f"masking_type: {self.masking_type} not supported with MOD")

    def _get_backbone(self):
        self.mm_pretrained = len(self.config.get('pretrain_modalities', [None])) > 1
        printlog(f'Pretrained model is_multimodal: {self.mm_pretrained}')
        if self.mm_pretrained:
            printlog(f'Pretraining modalities were "{self.config["pretrain_modalities"]}"')
            finetune_modalities = self.modalities

        self.use_all_seq = self.config['backbone_settings'].get('use_all_seq', False)

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

            elif self.phase in ['finetuning', 'scratch', 'linear_probing'] and len(self.modalities) == 2:
                self.backbone = mvit_finetune_multi_modality(out_features=self.num_classes,
                                                             pretrained=self.backbone_pretrained,
                                                             backbone=self.backbone_name,
                                                             patch_embeders=self.config['patch_embeders'],
                                                             **self.config['backbone_settings'],
                                                             modalities=self.modalities)

            elif self.phase == 'linear_probing' and len(self.modalities) == 1:
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

    def get_linear_probing(self, use_bn):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal -- assumes single modality finetune"""
        modality_token_names = [f"{modality}_token" for modality in self.modalities]
        if self.external_checkpoint_path is not None and (len(self.modalities) == 2) \
                and self.config['backbone_settings'].get('use_modality_token', False):
            printlog(
                f"NOT Freezing the modality tokens {modality_token_names} in the encoder for multimodal linear probing")
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
        printlog(
            f"Going to use linear probing --> trainable parameters: {n_trainable_parameters} -- frozen parameters: {n_frozen_parameters}")

        # re-init head
        printlog(f"re-init head with trunc_normal")

        if self.phase == 'linear_probing' and len(self.modalities) == 1:
            # single modality finetuning of multimodal-y pretrained model
            trunc_normal_(self.backbone.heads[self.modalities[0]].weight, std=0.01)
            if use_bn:
                self.backbone.heads[self.modalities[0]] = \
                    nn.Sequential(torch.nn.Sequential(
                        torch.nn.BatchNorm1d(self.backbone.heads[self.modalities[0]].in_features, affine=False,
                                             eps=1e-6),
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

    def fix_visible_tokens(self, num_visible_tokens_per_modality: torch.Tensor):
        if not (num_visible_tokens_per_modality.sum(dim=1) == self.num_visible_tokens).all():
            issue_index = torch.where(num_visible_tokens_per_modality.sum(dim=1) != self.num_visible_tokens)
            printlog(f"issue_index: {issue_index}")
            for i in issue_index:
                printlog(f"Fixing issue: num_visible_tokens_per_modality: {num_visible_tokens_per_modality[i]} "
                         f"does not sum to self.num_visible_tokens: {self.num_visible_tokens}")
                s = num_visible_tokens_per_modality[i].sum()
                diff = s - self.num_visible_tokens
                if diff > 0:
                    num_visible_tokens_per_modality[i.item()][0] -= diff

                else:
                    num_visible_tokens_per_modality[i.item()][0] += -diff

                printlog(f"Fixed issue: num_visible_tokens_per_modality: {num_visible_tokens_per_modality[i]} "
                         f"now sums to self.num_visible_tokens: {self.num_visible_tokens}")

    def get_masks(self, B, L, device, modality=None):
        if self.use_all_seq and self.masking_type == 'dirichlet':
            # ratios corresponds to the percentages of the masked tokens from each modality
            ratios = self.distr.sample((B,)).to(device)  # (B, num_modalities) sample on the fly for each batch (sum(dim=1)=1)

            # note: we assume L = Sum(L_i) for all modalities i and L_i is the same for all modalities
            L_modality = L / len(self.modalities)
            num_visible_tokens_per_modality = (ratios * self.num_visible_tokens).round().long()  # (B, num_modalities)

            # num_visible_tokens_per_modality = torch.tensor([[54, 44], [3, 95], [93, 5], [1, 97]], device=device)
            # check that num_visible_tokens_per_modality.sum(dim=1) == self.num_visible_tokens for all batch elements
            self.fix_visible_tokens(num_visible_tokens_per_modality)

            # masked_ratios corresponds to the percentages of the tokens per modality sequence that are masked
            masked_ratios = 1 - num_visible_tokens_per_modality / L_modality  # (B, num_modalities)
            mask_dict_batched = {"masked_ratios": masked_ratios, "num_visible_tokens_per_modality": num_visible_tokens_per_modality}

            # printlog(f"masked_ratios: {mask_dict_batched['masked_ratios']} num_visible_tokens_per_modality: {mask_dict_batched['num_visible_tokens_per_modality']}")
            sequence_length_per_modality = {m: self.backbone.encoder.patch_embeders[m].num_patches
                                            for m in self.modalities}

            masks_per_modality = []  # list of (B,L_m) masks
            ids_restore_per_modality = []  # list of (B,L_m)
            ids_restore_offset = 0
            for i, m in enumerate(self.modalities):
                L_m = sequence_length_per_modality[m]
                noise = torch.rand(B, L_m, device=device)  # (B, L_m)
                ids_shuffle = torch.argsort(noise, dim=1)  # sort ascending: high values are removed
                # note: ids_restore contains the permutation (per batch element) that if applied to the sequence
                # concat(tokens, masked_tokens) would restore its original order
                ids_restore = torch.argsort(ids_shuffle, dim=1)
                mask = torch.arange(L_m, device=device).unsqueeze(0).expand(B, -1)
                mask = torch.gather(mask, dim=1, index=ids_shuffle)
                mask = torch.where(mask < num_visible_tokens_per_modality[:, i].unsqueeze(1), 0, 1)
                mask_dict_batched[f'mask_{m}'] = mask
                ids_restore_per_modality.append(ids_restore + ids_restore_offset)
                ids_restore_offset += L_m

            mask_dict_batched['mask'] = torch.cat([mask_dict_batched[f'mask_{m}'] for m in self.modalities], dim=1)
            mask_dict_batched['ids_restore'] = torch.cat(ids_restore_per_modality, dim=1)
            mask_dict_batched['num_masked_tokens_per_modality'] = int(L_modality) - num_visible_tokens_per_modality
            mask_dict_batched['num_patches_per_modality'] = [L_modality] * B
            return mask_dict_batched
        else:
            raise NotImplementedError(f"masking_type: {self.masking_type} not supported with MOD")

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
