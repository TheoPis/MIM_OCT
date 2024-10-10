#!/usr/bin/env python3
# This is a slim implementation of the MAE model variants (MAE, MultiModalMAE)
# Instances of those classes are treated as the backbone attribute of more inflated classes (ViTMAE, MultiVitMae)
# that include configs and hyperparams.
# ViT_config.py is where instances are created and their different components are initialized

from typing import Union, List, Dict
import torch
import torch.nn as nn
from utils import printlog
from models.vit.ViT import DecoderCrossAttention
from models.vit.ViT_layers import AttentionPoolingClassifierMultimodal


class MAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: Union[None, nn.Module], head: nn.Module):
        """
        A model composed of an encoder (ViT) a decoder and a prediction head
        :param encoder: (VisionTransformer) ViT encoder
        :param head: output head
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.head = head

    @property
    def patch_size(self):
        # equivalent of patch_size_spatial
        # patch_embed.patch_size is (p_t, p_h, p_w) or (p_h, p_w) and always has square patches (p_h == p_w)
        return self.encoder.patch_embed.patch_size[1]  # p_w or p_h

    @property
    def patch_size_temporal(self):
        assert len(self.encoder.patch_embed.patch_size) == 3, \
            f"when asking for temporal patch_size it must be " \
            f"a tuple of 3 elements instead got {self.encoder.patch_embed.patch_size}"
        return self.encoder.patch_embed.patch_size[0]  # p_t

    @property
    def patch_size_spatial(self):
        return self.patch_size

    @property
    def first_patch_idx(self):
        # index of first token of sequence (i.e. after the optional cls_token)
        # if no cls_token or other non-sequence tokens (e.g. modality_token) this always returns 0)
        return self.encoder.first_patch_idx

    def forward(self, x, mask=None):
        # x: A tensor of shape [B,C,H,W] for images and [B,C,T,H,W] for videos
        # mask: A boolean tensor of the shape [B, patch_layout's shape]
        if mask is not None:
            # mae training
            return self.forward_masked(x, mask)
        else:
            return self.forward_standard(x)

    def forward_masked(self, x, mask_dict):
        # x.shape  # (B, C, H, W) -> (B, L, D)
        # tokenize, mask and forward encoder
        enc_outs = self.encoder(x, mask=mask_dict)  # x ~ (B, L', D)  L' < L
        x = enc_outs['emb']  # (B, L', D)
        # decoder_embed x, insert mask_tokens, add pos_embed and forward decoder
        x = self.decoder(x, mask_dict)
        x = x[:, self.first_patch_idx:]  # remove cls_token
        # reconstruct
        out = self.head(x)
        return out

    def forward_standard(self, x):
        # standard training (typically for supervised finetuning)
        enc_outs = self.encoder(x)
        out = self.head(enc_outs['emb'])
        return out


class MultiModalMAE(nn.Module):
    eligible_modalities = ['OCT', 'FA', 'IR', 'OCT_1', 'OCT_2', 'FA_1', 'FA_2']

    def __init__(self,
                 encoder: nn.Module,
                 decoders: Union[nn.ModuleDict, None],
                 heads: Union[nn.ModuleDict, None],
                 modalities: List[str]):
        """
        A model composed of a shared ViT encoder with multiple patch_embeders (one per modality)
        multiple decoders (one per modality) and heads (one per modality)
        Function:
                for some (m,x_m): emb = enc(x_m) and rec_x_m = head_m(dec_m([emb, masked_tokens_m]))
        :param encoder: (MultimodalVisionTransformer) ViT encoder shared by all modalities
        :param heads: list of heads, one for each modality
        :param modalities: list of modalities (strings), one for each decoder and head
        """
        super().__init__()
        self.encoder = encoder
        self.decoders = decoders
        self.heads = heads
        self.modalities = modalities
        # sanity checks
        assert all([modality in self.eligible_modalities for modality in self.modalities]), \
            f"modalities must be one of {self.eligible_modalities} instead got {self.modalities}"

        if self.decoders is not None:
            assert len(self.decoders) == len(self.heads) == len(self.modalities), \
                "number of decoders, heads and modalities must be the same instead got " \
                f"{len(self.decoders)}, {len(self.heads)} and {len(self.modalities)}"

    @property
    def first_patch_idx(self):
        # index of first token of sequence (i.e. after the optional cls_token)
        # if no cls_token or other non-sequence tokens (e.g. modality_token) this always returns 0)
        return self.encoder.first_patch_idx

    @property
    def patch_size(self):
        # assume all modalities have the same patch_size
        # assume patch are always squares and patch_size is a tuple so we index the first
        return self.encoder.patch_embeders[self.modalities[0]].patch_size[0]

    def forward(self,
                x: torch.Tensor,
                mask: Union[Dict[str, torch.Tensor], None] = None,
                modality: str = 'OCT')\
            -> [torch.Tensor, torch.Tensor]:
        # x: A tensor of shape [B,C,H,W] for images and [B,C,T,H,W] for videos
        # mask: A boolean tensor of the shape [B, patch_layout's shape]

        assert modality in self.modalities, f"modality {modality} not in {self.modalities}"
        if mask is not None:
            # mae training
            return self.forward_masked(x, mask, modality)
        else:
            return self.forward_standard(x, modality)

    def forward_masked(self, x, mask_dict, modality):
        orig_input_shape = x.shape  # (B, C, H, W) or (B, C, T, H, W)
        # mask and encode
        enc_outs = self.encoder(x, modality, mask_dict)  # x ~ (B, L', D)
        x = enc_outs['emb']  # (B, L', D)
        # decode with modality-specific decoder
        # decoder_embed x, insert mask_tokens, add pos_embed and forward decoder
        x = x[:, self.first_patch_idx:]  # remove cls_token/modality_token if used
        x = self.decoders[modality](x, mask_dict, modality=modality)
        # x = x[:, self.first_patch_idx:]  # remove cls_token/modality_token if used
        # reconstruct with modality-specific head
        out = self.heads[modality](x)
        return out

    def forward_standard(self, x, modality):
        # standard training (typically for supervised finetuning)
        enc_outs = self.encoder(x, modality)
        out = self.heads[modality](enc_outs['emb'])
        return out


class MultiModalMAESeq(nn.Module):
    eligible_modalities = ['OCT', 'FA', 'IR', 'OCT_1', 'OCT_2', 'FA_1', 'FA_2']

    def __init__(self,
                 encoder: nn.Module,
                 decoder: Union[nn.Module, None],
                 heads: Union[nn.ModuleDict, nn.Module, None],
                 modalities: List[str]):
        """
        A model composed of a shared ViT encoder that receives all modalities as a sequence
                            a decoder
                            a head
        Function:
                for some Seq = [(m,x_m) for m in Modalities]: emb = enc(Seq) and rec_Seq = head( dec(emb) )
        :param encoder: (MultimodalVisionTransformer) ViT encoder shared by all modalities
        :param heads: a dict of heads, one for each modality or a single head for finetuning tasks with Many-to-1 format
        :param modalities: list of modalities (strings), one for each decoder and head
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.heads = heads
        self.modalities = modalities
        # sanity checks
        assert all([modality in self.eligible_modalities for modality in self.modalities]), \
            f"modalities must be one of {self.eligible_modalities} instead got {self.modalities}"

    @property
    def first_patch_idx(self):
        # index of first token of sequence (i.e. after the optional cls_token)
        # if no cls_token or other non-sequence tokens (e.g. modality_token) this always returns 0)
        return self.encoder.first_patch_idx

    @property
    def patch_size(self):
        # assume all modalities have the same patch_size
        # assume patch are always squares and patch_size is a tuple so we index the first
        return self.encoder.patch_embeders[self.modalities[0]].patch_size[0]

    @property
    def use_modality_token(self):
        return self.encoder.use_modality_token

    def forward(self,
                x,
                mask,
                modality,
                return_embeddings=False)\
            -> [torch.Tensor, torch.Tensor]:
        # x: A tensor of shape [B,C,H,W] for images and [B,C,T,H,W] for videos
        # mask: A boolean tensor of the shape [B, patch_layout's shape]
        if mask is not None:
            # mae training
            return self.forward_masked(x, mask)
        else:
            if modality is not None:
                # todo this depends on the downstream task
                return self.forward_standard(x)
            else:
                return self.forward_multimodal_many_to_one(x, return_embeddings=return_embeddings)

    def get_tokens_from_modality(self, modality, num_kept_tokens):
        # todo
        pass

    def forward_masked(self, x: Dict[str, torch.Tensor], mask: Dict):
        # orig_input_shape = x.shape  # (B, C, H, W) or (B, C, T, H, W)
        # mask and encode
        enc_outs = self.encoder(x, mask_dict=mask)  # x ~ (B, L', D)
        x = enc_outs['emb']  # (B, L', D)
        B = x.shape[0]
        # info needed to insert_masked_tokens in the seq
        num_patches_per_modality = torch.Tensor([self.encoder.patch_embeders[m].num_patches
                                                 for m in self.modalities]).long().to(x.device)
        mask.update({f'num_patches_per_modality': num_patches_per_modality})

        # decoder_embed, insert_masked_tokens, decode
        if isinstance(self.decoder, nn.ModuleDict):
            # case1: one decoder per modality (no cross-modality attention)
            # note: the sequence is composed of all modalities concatenated and is of the following format:
            #       [cls_token, x1_m1, x2_m1, ..., xN_m1, x1_m2, x2_m2, ..., xK_m2]
            # N: num tokens from m1, K: num tokens from m2. N and K are the same for every example in the batch
            outs = {}  # to store the output for each modality
            x = x[:, self.first_patch_idx:]  # remove cls_token

            if all([not isinstance(dec, DecoderCrossAttention) for dec in self.decoder.values()]):
                # case1.1a: decoders without cross-modality attention num_keep_m constant across the batch
                if all([dec.batch_masking == 'symmetric' for dec in self.decoder.values()]):
                    start = 0
                    for m in self.modalities:
                        mask_m = {'mask': mask[f'mask_{m}']}
                        num_kept_m = (1.0-mask_m['mask']).sum(dim=1)[0].long().item()
                        x_m = (x[:, start:start + num_kept_m])  # x_m (B, L_m, D)
                        start += num_kept_m  # update starting index for next modality
                        x_m = self.decoder[m](x_m, mask_dict=mask_m, modality=m)
                        outs[m] = self.heads[m](x_m)
                    return outs
                else:
                    # case1.1b: decoders without cross-modality attention but with num_keep_m varies across the batch
                    # note implementation has some redundant computations but skips looping over the batch which is slow
                    start = 0
                    for m in self.modalities:
                        # project whole sequence to decoder dim
                        x_m = self.decoder[m].decoder_embed(x)
                        # insert modality masked tokens to whole sequence (we pass the complete mask)
                        x_m = self.decoder[m].insert_masked_tokens(x_m, mask['mask'], modality=m)
                        # crop only the tokens of the current modality
                        L_m = self.encoder.patch_embeders[m].num_patches
                        x_m = x_m[:, start:start + L_m]
                        start += L_m  # update starting index for next modality
                        # decode tokens from that modality
                        x_m = self.decoder[m](x_m, mask_dict=None, modality=m)
                        outs[m] = self.heads[m](x_m)
                    return outs
            else:
                # case1.2: decoders with cross-modality attention
                # for each modality
                #     slice its tokens from x (x_m)
                #     treat the rest as context (context)
                #     for each other modality
                #         get modality token from its decoder and repeat it to match kept_tokens per modality
                #         get pos_embed from decoder and add to context
                #     cnxt_modality_tokens = [modality_token_m1+pe, modality_token_m1+pe,..., modality_token_m2+pe, ...]
                #                            |------------------------------------|          |---------------------|
                #                                   num_kept_tokens_m1                         num_kept_tokens_m2 times
                #     pass x_m, context, context_modality_tokens through the modality decoder

                start = 0
                for m in self.modalities:
                    # fixme only works for 2 modalities !!!
                    other_modalities = [m_ for m_ in self.modalities if m_ != m]
                    mask_m = {'mask': mask[f'mask_{m}']}
                    # num_kept_m = (1.0-mask_m['mask']).sum(dim=1)[0].long().item()
                    num_kept_m = mask[f'num_keep_{m}']
                    x_m = (x[:, start: start+num_kept_m])  # x_m (B, L_m, D)
                    start += num_kept_m  # update starting index for next modality
                    context = x[:, num_kept_m:]  # context (B, L_c, D)
                    context_extra_tokens = []
                    for m_ in other_modalities:
                        mask_m_ = {'mask': mask[f'mask_{m_}']}
                        # assumption: all image have the same number of kept tokens per modality
                        num_kept_m_ = mask[f'num_keep_{m_}']
                        context_pos_emb = self.decoder[m].pos_embed.repeat(B, 1, 1)
                        # slice pos_emb at positions of kept tokens
                        kept_ids = torch.nonzero(1.0-mask_m_['mask'], as_tuple=False)
                        context_pos_emb = context_pos_emb[kept_ids[:, 0], kept_ids[:, 1], :].reshape(B, num_kept_m_, -1)
                        context_modality_tokens = getattr(self.decoder[m_], f"{m_}_token").repeat(B, num_kept_m_, 1)
                        context_extra_tokens.append(context_modality_tokens + context_pos_emb)

                    context_extra_tokens = torch.cat(context_extra_tokens, dim=1)
                    x_m = self.decoder[m](x_m, mask_dict=mask_m, modality=m, context=context,
                                          context_embeddings=context_extra_tokens)

                    outs[m] = self.heads[m](x_m)
                return outs

        else:
            # case2: shared decoder for all modalities
            x = self.decoder(x, mask_dict=mask)
            x = x[:, self.first_patch_idx:]  # remove cls_token if used
            outs = {}
            start = 0
            for m in self.modalities:
                end = start + self.encoder.patch_embeders[m].num_patches
                outs[m] = self.heads[m](x[:, start:end])
                start = end
            return outs

    def forward_multimodal_many_to_one(self, x, return_embeddings=False):
        enc_outs = self.encoder(x)  # x ~ (B, L', D)
        x = enc_outs['emb']  # [(B, D), (B,D)] (e.g with 2 modalities)
        if self.encoder.classifier_feature == 'global_pool_per_modality':
            x = torch.cat(x, dim=-1)  # (B,2D)
        elif self.encoder.classifier_feature == 'mean_max_pool_per_modality':
            x = torch.cat(x, dim=-1)  # (B,4D)  4D = mean + max for each modality
        elif self.encoder.classifier_feature in ['global_pool_from_modality', 'mean_max_pool_from_modality']:
            # todo hardcoded behaviour: assumes OCT is x[0] and IR is x[1]
            #  we default to OCT for this type of classifier
            x = x[0]  # keep only OCT embeddings
        if isinstance(self.heads, AttentionPoolingClassifierMultimodal):
            tokens_per_modality = {m:self.encoder.patch_embeders[m].num_patches for m in self.encoder.modalities}
            out = self.heads(x, tokens_per_modality)
        else:
            out = self.heads(x)  # note: self.heads is not a dict here todo improve clarity
        if return_embeddings:
            return out, enc_outs
        return out

    def forward_standard(self, x):
        raise NotImplementedError(f"forward_standard is not implemented yet")


# todo this is currently depracated and does not work with the rest of the code
class MultiModalMAECM(nn.Module):
    eligible_modalities = ['OCT', 'FA', 'IR', 'OCT_1', 'OCT_2', 'FA_1', 'FA_2']
    eligible_tasks = ['ssl', 'detection', 'segmentation']

    def __init__(self,
                 encoders: nn.ModuleDict,
                 encoders_cm: Union[nn.ModuleDict, None],
                 decoders: Union[nn.ModuleDict, None],
                 heads: Union[nn.ModuleDict, None],
                 modalities: List[str],
                 task: str = 'ssl'):
        """
        MultiModal MAE with Separated Encoders, Decoders, Heads
        A model composed of:
            - per-modality ViT encoders (contains per-modality patch_embedders)
            - per-modality CrossModal ViT encoders with SA-CA-MLP layers (CA: with context from other modality)
            - per-modality CrossAttention decoders (CAttn context is the output of the encoder for a different modality)
            - per-modality heads
        :param encoders(str:VisionTransformer): ViT encoder shared by all modalities [str, VisionTransformer]
        :param encoders_cm: (optional)(str:CrossModalTransformer) CrossModal ViT encoder shared by all modalities
        :param decoders: decoders (str:DecoderCrossAttention), one for each modality
        :param heads: list of heads, one for each modality [str, nn.Module]
        :param modalities: list of modalities (strings), one for each decoder and head
        """
        super().__init__()
        self.encoders = encoders
        self.decoders = decoders
        self.encoders_cm = encoders_cm
        self.grad_for_context_encoder = False  # todo make this an init argument: if true then learn context encoder
        self.heads = heads
        self.modalities = modalities
        self.task = task
        # sanity checks
        assert all([modality in self.eligible_modalities for modality in self.modalities]), \
            f"modalities must be one of {self.eligible_modalities} instead got {self.modalities}"

        if self.decoders is not None:
            assert len(self.encoders) == len(self.heads) == len(self.modalities) == len(self.decoders), \
                "number of encoders, decoders, heads and modalities must be the same instead got " \
                f"{len(self.encoders)}, {len(self.decoders)}, {len(self.heads)} and {len(self.modalities)}"

    @property
    def patch_size(self):
        # assume all modalities have the same patch_size
        # assume patch are always squares and patch_size is a tuple so we index the first
        return self.encoders[self.modalities[0]].patch_embed.patch_size[0]

    def forward(self, x: torch.Tensor, modality: str,
                x_m: torch.Tensor = None, modality_m: str = None, mask: Union[Dict[str, torch.Tensor], None] = None) \
            -> [torch.Tensor, torch.Tensor]:

        if self.decoders is not None:
            cls_token, x = self.forward_pretrain(x_m, x, modality_m, modality, mask)
            return cls_token, x

        else:
            x = self.forward_finetune(x, modality)
            return x

    def forward_finetune(self, x: torch.Tensor, modality: str):
        # note: assumes that the encoder is a VisionTransformer with classifier_feature = 'global_pool'
        if self.task in ['ssl', 'detection']:
            x = self.encoders[modality](x)
            x = self.heads[modality](x)
        elif self.task == 'segmentation':  # assume only the encoder is used here
            x = self.encoders[modality](x)  # here self.encoders[modality] is a ViTDet/ViT
        return x

    def forward_train(self, x_m: torch.Tensor, x: torch.Tensor, modality_m: str, modality: str,
                      mask: Union[Dict[str, torch.Tensor], None] = None) -> [torch.Tensor, torch.Tensor]:
        """
        :param x_m: A tensor of shape [N,C,H,W] that will be masked
        :param x: A tensor of shape [N,C,H,W] that will NOT be masked
        :param modality_m: name of the masked modality
        :param modality: name of the non-masked modality
        :param mask: (optional) a dictionary containing a mask tensor and ids_restore (indices of masked patches)
                    if None then model acts as a single modality ViT model
                 mask['mask_tensor']: A boolean tensor of the shape [N, trunk.patch_embedders[0].patch_layout's shape]
        :return: head output x and mask
        """
        printlog(f"Visible:{modality} Masked:{modality_m}")
        assert 'ids_restore' in mask.keys(), f'mask {mask} must contain ids_restore'
        assert modality in self.modalities, f"modality must be one of {self.modalities}"
        assert modality_m in self.modalities, f"modality_m must be one of {self.modalities}"""

        orig_input_shape = x.shape
        # encode with masking
        x_m = self.encoders[modality_m](x_m, mask=mask, return_features=True)

        # encode without masking (used as context)
        if self.grad_for_context_encoder:
            x = self.encoders[modality](x, return_features=True)  # this returns the whole sequence x (B,L+[1],C)
        else:
            with torch.no_grad():
                x = self.encoders[modality](x, return_features=True)

        # cross-modal encoder with context from other modality
        # also restores masked patches
        if isinstance(self.encoders_cm, nn.ModuleDict):
            x_m = self.encoders_cm[modality_m](x_m, context=x, mask=mask)

        # x_m = self.encoders[modality_m].insert_masked_tokens(x_m, mask['ids_restore'])

        # Parameter at index 527 with name backbone.heads.FA.bias

        # decoder/head to reconstruct masked patches (with CA context from other modality)
        x_m = self.decoders[modality_m](x_m, context=x, orig_input_shape=orig_input_shape)
        # input_pos_embed=self.encoders[modality_m].pos_embed)
        # Parameter at index 527 with name backbone.heads.FA.bias

        x_m = self.heads[modality_m](x_m)

        if self.encoders[modality].first_patch_idx == 1:
            # remove cls token
            cls_token = x_m[:, 0]
        else:
            cls_token = None
        return cls_token, x_m[:, 1:]
