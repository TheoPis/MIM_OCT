from models.mvit.MViT import MultimodalVisionTransformer
from typing import Union, Dict, List
import torch
from abc import ABC


class MultimodalVisionTransformerSeq(MultimodalVisionTransformer, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_modality_token and self.use_cls_token:
            self.first_patch_idx = 1

    def split_sequence(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        """ Gets a sequence [cls_token][x_m0][x_m1]...[x_mn]
        and returns cls_token, [x_m0,...,x_mn]
        :param x: sequence of shape (B, L, D) formatted as above
        :return:cls_token, patches
        """
        cls_token = x[:, : self.first_patch_idx]  # if first_patch_idx == 0, cls_token is (B,0,D) (treated as empty)
        x = x[:, self.first_patch_idx:]  # remove cls_token
        return cls_token, x

    def remove_masked_tokens(self, x, mask, **kwargs):
        """ remove masked tokens from the input
        :param x: sequence of shape (B, L, D) formatted as:
               x = [cls_token] [modality_token_0] [x_m0] [modality_token_1] [x_m1] ... [modality_token_n] [x_mn]
        :param mask: (B, L)
        :return: x (B, num_patches_kept, D)
        """
        B, D = x.shape[0], x.shape[-1]
        cls_token, patches = self.split_sequence(x)
        num_patches = patches.shape[1]
        # patches (B, L, D), mask (B, L) : patches[¬mask] (B*num_patches_kept, D) -> (B, num_patches_kept, D)
        patches = patches[~mask.to(torch.bool)].reshape(B, -1, D)
        # patches (B, num_patches_kept, D)
        x = torch.cat([cls_token, patches], dim=1)  # patches (B, [1]+num_patches_kept, C)
        return x

    def tokenize(self, x, mask_dict: Union[dict, None] = None, **kwargs):
        input_shape = x[list(x.keys())[0]].shape
        B = input_shape[0]

        seq = []
        npatch_per_img = []

        # seq is of the following form:
        # [cls_token, modality_token_1, tokens_modality_1, ..., modality_token_n, tokens_modality_n]
        # where tokens_modality_i is of shape (B, L_i, C), L_i is the number of patches for modality i
        if self.cls_token is not None:
            class_tokens = self.cls_token.expand(B, -1, -1)
            seq = [class_tokens] + seq

        for modality in x.keys():
            x_m = x[modality]
            x_m = self.patch_embeders[modality](x_m)  # add pos_embed to seq

            npatch_per_img.append(x_m.shape[1])

            # resize pos_embed in case of different sequence length
            # assume same across modalities
            pos_embed = self.get_pos_embedding(
                npatch_per_img=npatch_per_img[-1],
                pos_embed=self.pos_embed,
                patches_layout=self.patch_embeders[modality].patches_layout,
                input_shape=input_shape)

            x_m = x_m + pos_embed
            if self.use_modality_token:
                modality_token = getattr(self, f'{modality}_token').expand(B, -1, -1)
                x_m = x_m + modality_token
            seq.append(x_m)

        # create sequence
        seq = torch.cat(seq, dim=1)
        if self.masked_image_modeling and mask_dict is not None:
            seq = self.remove_masked_tokens(seq, mask_dict['mask'],
                                            num_masked_tokens_per_modality=mask_dict.get('num_masked_tokens_per_modality', None))
        # x = self.pos_drop(x)
        return seq, pos_embed

    def forward_features(self, x,  modality: str, mask_dict: dict = None, return_features=False)\
            -> Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
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
        x, pos_embed = self.tokenize(x, mask_dict)

        # encoder attention blocks
        for blk in self.blocks:
            x = blk(x)
        # x ~ (B, [2] + L, C)

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
            x = x[:, self.first_patch_idx:, ...].mean(dim=1)
            outs['emb'] = x
            return outs

        elif self.classifier_feature in ["global_pool_per_modality", "global_pool_from_modality"] \
                and not self.masked_image_modeling and not return_features:
            # case where we apply pooling over the last feature maps of encoder to perform classification
            start = self.first_patch_idx
            embs = []
            for m in self.modalities:
                L_m = self.patch_embeders[m].num_patches
                end = start + L_m
                embs.append(x[:, start:end, ...].mean(dim=1))
                start = end
            outs['emb'] = embs
            return outs
        elif self.classifier_feature in ["mean_max_pool_per_modality", "mean_max_pool_from_modality"] \
                and not self.masked_image_modeling and not return_features:
            # case where we apply mean and max pooling over tokens of each modality
            start = self.first_patch_idx
            embs = []
            for m in self.modalities:
                L_m = self.patch_embeders[m].num_patches
                end = start + L_m
                x_mean = x[:, start:end, ...].mean(dim=1)
                x_max, _ = torch.adaptive_max_pool1d(x[:, start:end, ...].transpose(2, 1), 1)
                embs.append(torch.cat([x_mean, x_max.squeeze(-1)], dim=1))
                start = end
            outs['emb'] = embs
            return outs

        else:
            raise NotImplementedError(f"this mode is not implemented "
                                      f"classifier_feature: {self.classifier_feature}, "
                                      f"mim: {self.masked_image_modeling},"
                                      f"return_features: {return_features}"
                                      f"mask_dict: {mask_dict}")

    def forward(self,
                x: Dict[str, torch.Tensor],
                modality: str = None,
                mask_dict: Dict[str, torch.Tensor] = None,
                return_features=False) -> Dict[str, torch.Tensor]:
        """
        :param x: input
        :param modality: string indicating the modality
        :param mask_dict:
        :param return_features:
        :return: outs {'emb': x, 'pos_embed'(only when masked_image_modeling) : pos_embed}
        """
        if self.classifier_feature is None:
            return_features = True
        elif not self.masked_image_modeling:
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
