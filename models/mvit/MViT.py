#!/usr/bin/env python3
import torch
from torch import nn
from abc import ABC
from functools import partial
from typing import List, Dict, Union
from models.vit.ViT import VisionTransformerBase
from models.vit.ViT_layers import PatchEmbed, trunc_normal_, Block


class MultimodalVisionTransformer(VisionTransformerBase, ABC):
    """ ViT that supports multiple modalities with:
        - separate patch_embedders
        - a shared encoder
    Reuses most of the functionality from models/ViT.py
    """
    valid_classifier_features = ["cls_token", "global_pool", "global_pool_per_modality", "mean_max_pool",
                                 "global_pool_from_modality", "mean_max_pool_per_modality",
                                 "mean_max_pool_from_modality", None]

    def __init__(
        self,
        modalities: List[str],
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: int = 4,
        attn_target: partial = None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        classifier_feature=None,
        use_cls_token=True,
        use_modality_token=False,
        modality_token_mode='add',
        learnable_pos_embed=False,
        patch_embed_types: List[str] = None,
        patch_embed_params_list: Union[Dict[str, List[str]]] = None,
        layer_norm_eps: float = 1e-6,
        masked_image_modeling=False,
        patch_restore=False,  # if True then creates mask_tokens that are used to restore masked tokens
        mask_token_embed_dim=None,
    ):
        ############################################# sanity checks #############################################
        assert use_cls_token or classifier_feature in ["global_pool"]
        assert classifier_feature in self.valid_classifier_features, \
            f"classifier_feature '{classifier_feature}' not in {self.valid_classifier_features}"
        self.classifier_feature = classifier_feature
        if patch_embed_params_list is None:
            patch_embed_params_list = dict()
            for modality in modalities:
                patch_embed_params_list[modality] = [None]
        elif isinstance(patch_embed_params_list, dict):
            for modality in modalities:
                if len(patch_embed_params_list[modality]) == 0:
                    patch_embed_params_list[modality] = [None]
        else:
            raise ValueError(f"patch_embed_params_list must be a list or None instead got {patch_embed_params_list}")

        self.masked_image_modeling = masked_image_modeling  # switch that controls whether to train using masking
        # multimodal settings
        assert len(modalities) > 0, "At least one modality must be provided"
        self.num_modalities = len(modalities)
        self.modalities = modalities

        # check num_modalities equal to num of patch_embedders
        assert self.num_modalities == len(patch_embed_types), \
            f"Number of modalities must equal number of patch embedders " \
            f"{self.num_modalities} ({self.modalities}) != {len(patch_embed_types)}, ({patch_embed_types})"

        ############################################# patch_embed ######################################################
        patch_embeders = nn.ModuleDict()
        for patch_embed_type, modality in zip(patch_embed_types, modalities):
            if patch_embed_type == "linear":
                patch_embeders[modality] = PatchEmbed(img_size=img_size,
                                                      patch_size=patch_size,
                                                      in_chans=in_chans,
                                                      embed_dim=embed_dim,
                                                      pad_func=patch_embed_params_list[modality][0])
            else:
                raise ValueError(f"Unknown patch_embed_type {patch_embed_type}")

        # check: that each patch_embeder has the same num_patches (as the ViT encoder expects a fixed sequence length L)
        num_patches = patch_embeders[modalities[0]].num_patches
        for modality in modalities:
            assert patch_embeders[modality].num_patches == num_patches, \
                f"All patch embedders must have the same number of patches instead patch_embedder for {modality}" \
                f" has {patch_embeders[modality].num_patches} patches while " \
                f"patch_embedder for {modalities[0]} has {num_patches} patches"

        # temporal dimension
        # note: patches_layout is [p_t, p_h, p_w] (if p_t==1 then 2D data else 3D)
        if len(patch_embeders[self.modalities[0]].patches_layout) == 2:
            has_temporal_dim = False
        else:
            has_temporal_dim = patch_embeders[self.modalities[0]].patches_layout[0] > 1

        ################################# init mask_tokens, cls_token and pos_embed ####################################
        build_pos_embed = True
        super().__init__(num_patches, embed_dim, use_cls_token, mask_token_embed_dim, build_pos_embed, has_temporal_dim,
                         use_modality_token=use_modality_token, modalities=self.modalities,
                         modality_token_mode=modality_token_mode)
        self.patch_embeders = patch_embeders
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        self.num_features = self.embed_dim  # fixme unused
        self.pos_drop = nn.Dropout(p=drop_rate)

        ###################### attention blocks ########################################################################
        # stochastic depth decay
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(depth)]
        else:
            raise NotImplementedError(f"Drop path types are: [progressive, uniform]. Got {drop_path_type}.")

        # attention blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=None,
                    layer_scale_init_value=1e-4
                )
                for i in range(depth)
            ]
        )
        if self.masked_image_modeling:
            self.norm = norm_layer(embed_dim)  # normalization at the end of the encoder and before cls_head
        else:
            self.norm = torch.nn.Identity()

        self.patch_restore = patch_restore

        ###################### initialization of pos_embed, cls_token, modality_token and patch_embed ##################
        if learnable_pos_embed:  # this is not used as we fix the pos embed to be sinusoidal
            trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:  # init the cls_token is it being used
            trunc_normal_(self.cls_token, std=0.02)
        if use_modality_token:  # init the modality_token is it being used
            for modality in modalities:
                trunc_normal_(getattr(self, f"{modality}_token"), std=0.02)

        if self.masked_image_modeling:
            for i, modality in enumerate(modalities):
                if patch_embed_types[i] == "linear":
                    # Based on MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
                    w = self.patch_embeders[modality].proj.weight.data
                    torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.masked_image_modeling:  # Based on MAE and official Jax ViT implementation
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def tokenize(self, x, modality: str, mask_dict: Union[dict, None] = None):
        """ Prepare the input tokens for the transformer encoder: Embed patches and add cls_token
            x ~ (B, C, H, W) -> PatchEmbed(x) ~ (B, L, C) -> (B, L, C) + cls_token (B, 1, C) -> (B, 1+L, C)
        :param x: (B, C, H, W)
        :param modality:
        :param mask_dict:  optional dict with keys 'mask' and 'ids_restore'
        :return: x (B, [1]+L, C) (cls_token is added optionally)
        """
        B = x.shape[0]
        input_shape = x.shape

        seq = []
        # seq is going to be created with the following form:
        # [cls_token, modality_token_i, tokens_modality_i] where i alternates between modalities
        # or [cls_token, tokens_modality_i] if use_modality_token=True and modality_token_mode='add'

        if self.cls_token is not None:
            class_tokens = self.cls_token.expand(B, -1, -1)
            seq = [class_tokens] + seq

        x = self.patch_embeders[modality](x)
        num_patches_per_img = x.shape[1]  # L

        # resize pos_embed in case of different sequence length
        pos_embed = self.get_pos_embedding(
            npatch_per_img=num_patches_per_img,
            pos_embed=self.pos_embed,
            patches_layout=self.patch_embeders[modality].patches_layout,
            input_shape=input_shape,
            first_patch_idx=self.first_patch_idx,
            has_temporal_dim=self.has_temporal_dim
        )

        # add pos_embed to patch tokens
        x += pos_embed

        # add modality_token to patch tokens
        if self.use_modality_token:
            modality_token = getattr(self, f'{modality}_token').expand(B, -1, -1)
            if self.modality_token_mode == 'add':
                x = x + modality_token
            else:  # concat
                seq = [modality_token] + seq   # prepend modality_token to seq
        seq.append(x)

        # prepend pos_embed_cls to pos_embed
        seq = torch.cat(seq, dim=1)
        if self.masked_image_modeling and mask_dict is not None:
            seq = self.remove_masked_tokens(seq, mask_dict['mask'])
        # x = self.pos_drop(x)
        return seq, pos_embed

    def forward_features(self, x,  modality: str, mask_dict: dict = None, return_features=False):
        """ x ~ (B, C, H, W) -> PatchEmbed(x) ~ (B, L, C) -> (B, L, C) + [cls_token (B, 1, C)] -> (B, [1] + L, C)
        :param x: (B, C, H, W)
        :param modality: string indicating the modality
        :param mask_dict: optional dict with keys 'mask' and 'ids_restore'
        :param return_features:
        :return: x (B, [1]+L, C) if classifier_feat
        """
        outs = dict()
        # orig_input_shape = x.shape

        # modality-specific tokenize
        x, pos_embed = self.tokenize(x, modality, mask_dict)

        # encoder attention blocks
        for blk in self.blocks:
            x = blk(x)
        # x ~ (B, [1] + L, C)

        if mask_dict is not None and self.masked_image_modeling:
            x = self.norm(x)
            outs['emb'] = x
            outs['pos_embed'] = pos_embed  # need to pass this to the decoder
            return outs

        elif not self.masked_image_modeling and return_features:
            # case no masking, no decoder, no classifier, only get features
            x = self.norm(x)
            outs['emb'] = x
            return outs

        elif self.classifier_feature == "cls_token" and (mask_dict is None or self.decoders is None):
            # case where we perform classification using only the cls_token
            assert self.first_patch_idx == 1, "Must have a CLS token at 0"
            x = x[:, 0]  # take only cls_token
            outs['emb'] = x
            return outs

        elif self.classifier_feature == "global_pool" and not self.masked_image_modeling and not return_features:
            # case where we apply pooling over the last feature maps of encoder to perform classification
            x = x[:, self.first_patch_idx:, ...].mean(dim=1)
            outs['emb'] = x
            return outs

        elif self.classifier_feature == "mean_max_pool" and not self.masked_image_modeling and not return_features:
            # a la Kurmann
            # case where we apply pooling over the last feature maps of encoder to perform classification
            x_mean = x[:, self.first_patch_idx:, ...].mean(dim=1)
            x_max, _ = torch.adaptive_max_pool1d(x[:, self.first_patch_idx:, ...].transpose(2, 1), 1)
            outs['emb'] = torch.cat([x_mean, x_max.squeeze(-1)], dim=1)
            return outs

        else:
            raise NotImplementedError(f"this mode is not implemented "
                                      f"classifier_feature: {self.classifier_feature}, "
                                      f"mim: {self.masked_image_modeling}, "
                                      f"return_features: {return_features}, "
                                      f"mask_dict: {mask_dict}")

    def forward(self,
                x: torch.Tensor,
                modality: str,
                mask_dict: Dict[str, torch.Tensor] = None,
                return_features=False) -> Dict[str, torch.Tensor]:
        """
        :param x: input
        :param modality: string indicating the modality
        :param mask_dict:
        :param return_features:
        :return: outs {'emb': x, 'pos_embed'(only when masked_image_modeling) : pos_embed}
        """
        if not self.masked_image_modeling:
            assert return_features or self.classifier_feature in self.valid_classifier_features,\
                "If masked_image_modeling=False must have" \
                f" return_features=True, or classifier_feature='global_pool'" \
                f" instead got self.classifier_feature = {self.classifier_feature}," \
                f" return_features={return_features}"

        else:
            assert mask_dict is not None, f"if masked_image_modeling=True must provide a mask"

        assert (not self.masked_image_modeling) or (mask_dict is not None)
        outs = self.forward_features(x, mask_dict=mask_dict,  modality=modality, return_features=return_features)
        return outs
