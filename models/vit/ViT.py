#!/usr/bin/env python3
import math
from abc import ABC
from torch import Tensor
from utils import printlog
from .ViT_layers import *


class VisionTransformerBase(nn.Module, ABC):
    def __init__(self,
                 num_patches: int,  # number of patches obtained when patchify-ing the input data
                 embed_dim: int,  # dimension of the embedding
                 use_cls_token: bool,  # whether to use a cls_token
                 mask_token_embed_dim: Union[None, int] = None,
                 build_pos_embed: bool = True,  # whether to build positional embedding
                 has_temporal_dim: bool = False,  # whether input is 3D (False means it's to 2D)
                 learnable_pos_embed: bool = False,  # whether to learn positional embedding
                 use_modality_token: bool = False,  # whether to use a modality token
                 modalities: Union[None, List] = None,
                 use_all_seq: bool = False,  # whether multiple modalities are concatenated in the input sequence
                 modality_token_mode: str = 'add'  # how to use the modality token: 'concat' or 'add'
    ):
        """ Base class for Vision Transformer models with shared functionality for both Encoder and Decoder
        :param num_patches:  number of patches obtained when patchify-ing the input data
        :param embed_dim:  dimension of the embedding
        :param use_cls_token:  whether to use a cls_token
        :param mask_token_embed_dim:  dimension of the mask token embedding
        :param has_temporal_dim (default: False):  whether input is 3D (False means it's to 2D)
        :param learnable_pos_embed (default : False):  whether to learn positional embedding
        """
        super().__init__()
        ###################################### cls_token ###############################################################
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            # self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False) # legacy todo remove
            self.first_patch_idx = 1
            total_num_tokens = num_patches + 1
        else:
            self.cls_token = None
            # self.pos_embed_cls = torch.zeros(1, 0, embed_dim) # legacy todo remove
            self.first_patch_idx = 0
            total_num_tokens = num_patches

        ###################################### modality_token ##########################################################
        self.use_modality_token = use_modality_token   # this token is added so total_num_patches is not increased
        self.modality_token_mode = modality_token_mode  # this affects tokenize and insert_masked_tokens
        assert self.modality_token_mode in ['add', 'concat'], \
            f"modality_token_mode must be 'add' or 'concat' but given {self.modality_token_mode}"
        if self.use_modality_token and self.modality_token_mode == 'concat':
            self.first_patch_idx += 1
            total_num_tokens += 1
        printlog(f"setting use_modality_token: {self.use_modality_token} "
                 f"and modality_token_mode: {self.modality_token_mode} and first_patch_idx: {self.first_patch_idx} ")
        # build modality token
        if use_modality_token:
            assert modalities is not None
            for modality in modalities:
                modality_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
                self.register_parameter(f"{modality}_token", modality_token)
                printlog(f"initialized modality_token: {modality}_token of shape {modality_token.shape} in {self}")

        ###################################### mask_token ##############################################################
        if mask_token_embed_dim is not None:
            if modalities is None:
                self.mask_token = nn.Parameter(torch.zeros(1, 1, mask_token_embed_dim), requires_grad=True)
                trunc_normal_(self.mask_token, std=0.02)
            else:
                # initialize a dict of mask_tokens, one for each modality.
                self.mask_tokens = {}
                for modality in modalities:
                    # initialized to zeros following iBOT
                    mask_token = nn.Parameter(torch.zeros(1, mask_token_embed_dim), requires_grad=True)
                    self.mask_tokens[modality] = mask_token
                    self.register_parameter(f"{modality}_mask_token", self.mask_tokens[modality])
                    printlog(f"initialized {modality}_mask_token of shape {self.mask_tokens[modality].shape} in {self}")
        else:
            self.mask_token = None

        ###################################### pos_embed ###############################################################
        self.has_temporal_dim = has_temporal_dim
        if build_pos_embed:
            self.build_pos_embed(learnable_pos_embed, total_num_tokens, num_patches)

        ###################################### multimodal sequence #####################################################
        self.use_all_seq = use_all_seq  # whether multiple modalities are concatenated in the input sequence
        printlog(f"setting use_all_seq: {self.use_all_seq} in {self}")

    def build_pos_embed(self, learnable_pos_embed, total_num_tokens, num_patches):
        if learnable_pos_embed:
            if self.has_temporal_dim:
                raise NotImplementedError("Learnable pos embed not supported for video models")
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, total_num_tokens, self.embed_dim), requires_grad=True)
        else:
            if self.has_temporal_dim:
                # we use a seperable time/space embedding which are then added to get the pos_embed
                # following https://github.com/facebookresearch/mae_st/blob/main/models_mae.py#L42
                # note: we assume square space dims (e.g 224x224)
                self.register_buffer("pos_embed_temporal",
                                     get_sinusoid_encoding_table(self.patch_embed.patches_layout[0], self.embed_dim))
                self.register_buffer("pos_embed_spatial",
                                     get_sinusoid_encoding_table(np.prod(self.patch_embed.patches_layout[1:]),
                                                                 self.embed_dim))
            else:
                # note: use num_patches instead of total_num_patches as we don't want to add pos_embed to the cls token
                self.register_buffer("pos_embed", get_sinusoid_encoding_table(num_patches, self.embed_dim))
                printlog(f"built sinusoid positional embedding of shape {self.pos_embed.shape} in {self}")

    def remove_masked_tokens(self, x, mask):
        """ remove masked tokens from the input
        :param x: (B, L, D)
        :param mask: (B, L)
        :return: x (B, num_patches_kept, D)
        """
        B, D = x.shape[0], x.shape[-1]
        cls_token = x[:, : self.first_patch_idx]  # if first_patch_idx == 0, cls_token is (B,0,D) (treated as empty)
        patches = x[:, self.first_patch_idx:]
        # patches (B, L, D), mask (B, L) : patches[¬mask] (B*num_patches_kept, D) -> (B, num_patches_kept, D)
        patches = patches[~mask.to(torch.bool)].reshape(B, -1, D)
        # patches (B, num_patches_kept, D)
        x = torch.cat([cls_token, patches], dim=1)  # patches (B, [1]+num_patches_kept, C)
        return x

    def insert_masked_tokens_legacy(self, x, ids_restore, **kwargs):  # todo safe remove
        modality = kwargs.get("modality", None)
        B, num_patches_kept, D = x.shape
        sequence_length = ids_restore.shape[1]
        if self.first_patch_idx > 0:
            num_patches_kept -= self.first_patch_idx  # first tokens are cls and/or modality tokens not : masked tokens
        num_patches_masked = sequence_length - num_patches_kept  # num_patches_masked = L - num_patches_kept
        # repeat mask_token num_patches_masked times
        if modality is None:
            mask_tokens = self.mask_token.repeat(B, num_patches_masked, 1)  # mask_tokens (B, num_patches_masked, D)
        else:
            mask_tokens = self.mask_tokens[modality].repeat(B, num_patches_masked, 1)

        x_ = torch.cat([x[:, self.first_patch_idx:, :], mask_tokens], dim=1)  # remove cls token --> x_ (B, L, D)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # restore ordering
        # append cls and/or modality token if used (else nothing happens)
        x = torch.cat([x[:, :self.first_patch_idx, :], x_], dim=1)
        return x

    def insert_masked_tokens(self, x, mask, **kwargs):
        """ insert mask tokens in the sequence and then restore order. Works for Decoders of ViT or MViT

        :param x: unmasked tokens (B, [1] + num_patches_kept, D)
        :param mask: (B,  L), 1 means remove token, 0 means keep token
        :return: x (B, [1] + L, D)
        """

        L = mask.shape[1]
        modality = kwargs.get("modality", None)
        B, num_patches_kept, D = x.shape
        sequence_length = mask.shape[1]
        if self.first_patch_idx > 0:
            num_patches_kept -= self.first_patch_idx  # first tokens are cls and/or modality tokens not : masked tokens
        # num_patches_masked = sequence_length - num_patches_kept  # num_patches_masked = L - num_patches_kept
        # repeat mask_token num_patches_masked times
        if modality is None:
            # single modality training
            mask_token = self.mask_token.to(x.dtype)  # mask_tokens (B, num_patches_masked, D)
        elif hasattr(self, "mask_tokens"):
            # multi modality training with shared decoder
            mask_token = self.mask_tokens[modality].to(x.dtype)
        elif hasattr(self, f"{modality}_token"):
            # multi modality training with separate decoders
            mask_token = getattr(self, f"{modality}_token").to(x.dtype)
        else:
            raise ValueError(f"modality {modality} not found in {self}")

        mask = mask.view(B, -1)
        B, N = mask.shape
        tmp = torch.empty(B, N, D, dtype=x.dtype).to(x.device)
        tmp[mask.to(torch.bool)] = mask_token
        tmp[~mask.to(torch.bool)] = x[:, self.first_patch_idx:].reshape(-1, D)
        x = torch.cat([x[:, :self.first_patch_idx], tmp], dim=1)
        return x

    def insert_masked_tokens_seq(self, x, mask_dict, **kwargs):
        """ insert mask tokens in the sequence and then restore order. Works for a shared Decoder MViTSeq
        follows https://github.com/facebookresearch/mae/blob/main/models_mae.py
        :param x: unmasked tokens (B, [1] + num_patches_kept, D)
        :param mask_dict: (B, L), 1 means remove token, 0 means keep token
        :return: x (B, [1] + L, D)
        """
        ############################### sanity checks ##################################################################
        assert self.use_all_seq, f"insert_masked_tokens_seq only works for" \
                                 f" MViTSeq that has self.use_all_seq=True instead got {self}"
        num_masked_tokens_per_image_per_modality = mask_dict.get('num_masked_tokens_per_modality', None)
        num_patches_per_modality = mask_dict.get('num_patches_per_modality', None)
        assert num_masked_tokens_per_image_per_modality is not None, "num_masked_tokens_per_image_per_modality " \
                                                                     "is not in mask_dict"
        assert num_patches_per_modality is not None, "num_patches_per_modality is not in mask_dict"

        mask = mask_dict['mask']
        # sequence_length = mask.shape[1]
        B, num_patches_kept, D = x.shape
        num_patches_kept -= self.first_patch_idx  # first tokens are cls and/or modality tokens not : masked tokens
        # num_patches_masked = sequence_length - num_patches_kept  # num_patches_masked = L - num_patches_kept

        cls_token = x[:, :self.first_patch_idx, :]  # (B, 1, D)
        x = x[:, self.first_patch_idx:, :]  # (B, num_patches_kept, D)
        x_batch = []
        # start/end index x and start_mask indexes mask
        for b in range(B):
            start = 0
            start_mask = 0
            seq = []  # sequence of tokens for each element of the batch
            for i, m in enumerate(self.modalities):

                num_masked_tokens_m = num_masked_tokens_per_image_per_modality[b, i].item()
                num_patches_m = num_patches_per_modality[i].item()   # this is constant per modality across batch
                num_kept_tokens_m = num_patches_m - num_masked_tokens_m
                mask_m = mask[b, start_mask:start_mask + num_patches_m]  # (num_patches_m,)
                start_mask += num_patches_m

                tmp = torch.empty(num_patches_m, D, dtype=x.dtype).to(x.device)  # (num_patches_m, D)
                tmp[mask_m.to(torch.bool)] = self.mask_tokens[m].to(x.dtype)  # place mask tokens where masked
                end = start + num_kept_tokens_m
                tmp[~mask_m.to(torch.bool)] = x[b, start:end, :]  # place the visible tokens where not masked
                start = end

                seq.append(tmp)
            x_batch.append(torch.cat(seq, dim=0))  # cat across modalities in image b and append to batch
        x_batch = torch.stack(x_batch, dim=0)  # cat across elements of the batch

        x = torch.cat([cls_token, x_batch], dim=1)  # prepend cls token if used (else nothing happens)
        return x

    @classmethod
    def get_pos_embedding(
        cls,
        npatch_per_img: int,
        pos_embed: torch.Tensor,
        patches_layout: Union[list, tuple],
        input_shape=None,
        first_patch_idx=1,
        has_temporal_dim=False,
    ):

        pos_embed = cls.interpolate_pos_encoding(
            npatch_per_img,
            pos_embed,
            patches_layout,
            has_temporal_dim=has_temporal_dim
        )
        return pos_embed

    @classmethod
    def interpolate_pos_encoding(
        cls,
        num_patches_target,  # L or (T//p_t, H//p * W//p)
        pos_embed,  # 1, L, D
        patches_layout_target,  # (T//p_t, H//p, W//p)
        input_shape=None,  # (B,C,H,W)
        first_patch_idx=1,
        has_temporal_dim=False
    ):
        num_patches_prev = pos_embed.shape[1]
        if num_patches_target == num_patches_prev:
            return pos_embed

        if has_temporal_dim:
            # todo implement interpolation for spatio-temporal pos_embed
            raise NotImplementedError("temporal interpolation not supported yet : todo !")

        if patches_layout_target[0] == 1:
            # simple 2D pos embedding, no temporal component
            pos_embed = cls.interpolate_pos_encoding_2d(num_patches_target, pos_embed)
        elif patches_layout_target[0] > 1:
            # pos_embed has a temporal component
            # fixme
            raise NotImplementedError("temporal interpolation not supported yet : todo !")
        else:
            raise ValueError(f"unknown interpolation with patches_layout_target: {patches_layout_target}")
        return pos_embed

    @staticmethod
    def interpolate_pos_encoding_2d(target_L, pos_embed):
        """
        Interpolate positional embedding computed for L a new L'
        :param target_L: L'
        :param pos_embed: (1, L, D)
        :return: (1, L', D)
        :return: (1, L', D)
        """
        L = pos_embed.shape[1]
        if L == target_L:
            return pos_embed
        dim = pos_embed.shape[-1]

        # unflatten pos_embed
        pos_embed = pos_embed.reshape(1, int(math.sqrt(L)), int(math.sqrt(L)), dim)
        # pos_embed is (1, H//p, W//p, D) where H//p = W//p = sqrt(L)
        pos_embed = nn.functional.interpolate(
            pos_embed.permute(0, 3, 1, 2),  # (1, D, H//p, W//p)
            scale_factor=math.sqrt(target_L / L),
            mode="bicubic",
        )
        # pos_embed is (1, D, s * H//p, s * W//p) where s = sqrt(L' / L)
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)  # flatten to (1, L', D)
        return pos_embed

    @staticmethod
    def interpolate_pos_encoding_3d(target_L, pos_embed):
        # todo
        raise NotImplementedError("temporal interpolation not supported yet : todo !")

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


class VisionTransformer(VisionTransformerBase, ABC):
    """
    Vision transformer. Adding stochastic depth makes it a DeiT.
    """
    valid_classifier_features = ["cls_token", "global_pool", None, "mean_max_pool"]

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        mlp_ratio: int = 4,
        attn_target: partial = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        drop_path_type: str ="progressive",
        classifier_feature: Union[str, None] = None,
        use_cls_token: bool = True,
        learnable_pos_embed=False,
        patch_embed_type: str = "linear",
        patch_embed_params_list=None,
        layer_norm_eps=1e-6,
        masked_image_modeling=False,
        patch_restore=False,  # if True then creates mask_tokens that are used to restore masked tokens
        mask_token_embed_dim=None,
    ):

        ############################################# sanity checks ####################################################
        assert use_cls_token or classifier_feature in ["global_pool"]
        assert classifier_feature in self.valid_classifier_features, \
            f"classifier_feature '{classifier_feature}' not in {self.valid_classifier_features}"
        self.classifier_feature = classifier_feature
        if patch_embed_params_list is None:
            patch_embed_params_list = [None]
        elif isinstance(patch_embed_params_list, list):
            if len(patch_embed_params_list) == 0:
                patch_embed_params_list = [None]
        else:
            raise ValueError(f"patch_embed_params_list must be a list or None instead got {patch_embed_params_list}")
        self.masked_image_modeling = masked_image_modeling  # switch that controls whether to train using masking

        ############################################# patch_embed ######################################################
        if patch_embed_type == "linear":
            patch_embed = PatchEmbed(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    pad_func=patch_embed_params_list[0])
        else:
            raise ValueError(f"Unknown patch_embed_type {patch_embed_type}")

        num_patches = patch_embed.num_patches  # sequence length L
        assert (patch_embed.patches_layout[-1] == patch_embed.patches_layout[-2]),\
            "Interpolation of pos embed not supported for non-square layouts"

        # temporal dimension
        if len(patch_embed.patches_layout) == 2:
            has_temporal_dim = False
        else:
            has_temporal_dim = patch_embed.patches_layout[0] > 1

        ################################# init mask_token, cls_token and pos_embed #####################################
        build_pos_embed = True
        super().__init__(num_patches, embed_dim, use_cls_token, mask_token_embed_dim, build_pos_embed, has_temporal_dim)

        self.patch_embed = patch_embed
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
                    layer_scale_init_value=1e-4,
                )
                for i in range(depth)
            ]
        )

        if self.masked_image_modeling:
            self.norm = norm_layer(self.embed_dim)  # normalization at the end of the encoder
        else:
            self.norm = torch.nn.Identity()

        self.patch_restore = patch_restore

        ###################### initialization of pos_embed, cls_token and patch_embed ##################################
        if learnable_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            trunc_normal_(self.cls_token, std=0.02)
        if self.masked_image_modeling and patch_embed_type == "linear":
            # Based on MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
            w = self.patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # initialize weights
        self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def tokenize(self, x, mask_dict: Union[dict, None] = None, **kwargs):
        """ Prepare the input tokens for the transformer encoder: Embed patches and add cls_token
            x ~ (B, C, H, W)
            PatchEmbed(x) ~ (B, L, C) -> (B, L, C) + cls_token (B, 1, C) -> (B, 1+L, C)
            cls_token prepend (optional) [cls_token, x] ~ [(B,1,C), (B, L, C)] -> (B, L+[1], C)
            pos_embed addition ~ x + [pos_embed_cls, pos_embed] ~ (B, L+[1], C) + [(1, 1, C), (1,L,C)] -> (B, L+[1], C)
        :param x: (B, C, H, W)
        :param mask_dict:  optional dict with keys 'mask' and 'ids_restore'
        :return: x (B, [1]+L, C) (cls_token is added optionally)
        """
        B = x.shape[0]
        input_shape = x.shape

        x = self.patch_embed(x)
        num_patches_per_img = x.shape[1]  # L (without cls_token)

        if self.cls_token is not None:
            class_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((class_tokens, x), dim=1)

        if self.has_temporal_dim:  # separable spatio-temporal pos_embed
            # https://github.com/facebookresearch/mae_st/blob/main/models_mae.py
            # patches are (p_t, p_h, p_w)  = patch_embed.patches_layout
            num_temporal_patches = self.patch_embed.patches_layout[0]
            num_spatial_patches = np.prod(self.patch_embed.patches_layout[1:])
            pe_s = self.pos_embed_spatial.repeat(1, num_temporal_patches, 1)
            pe_t = torch.repeat_interleave(self.pos_embed_temporal, num_spatial_patches, dim=1)
            pos_embed = pe_s + pe_t
            pos_embed = self.get_pos_embedding(
                npatch_per_img=num_patches_per_img,
                pos_embed=pos_embed,
                patches_layout=self.patch_embed.patches_layout,
                input_shape=input_shape,
                first_patch_idx=self.first_patch_idx,
                has_temporal_dim=self.has_temporal_dim
            )

        else:
            pos_embed = self.get_pos_embedding(
                npatch_per_img=num_patches_per_img,
                pos_embed=self.pos_embed,
                patches_layout=self.patch_embed.patches_layout,
                has_temporal_dim=self.has_temporal_dim)

        x[:, self.first_patch_idx:] += pos_embed  # skip the cls_token when adding pos_embed
        if self.masked_image_modeling:
            assert mask_dict is not None, "got mask_dict None with for masked_image_modeling: True"
            x = self.remove_masked_tokens(x, mask_dict['mask'])
        # x = self.pos_drop(x)
        return x, pos_embed

    def forward_features(self, x, mask_dict=None, use_checkpoint=False, return_features=False):
        """ x ~ (B, C, H, W) -> PatchEmbed(x) ~ (B, L, C) -> (B, L, C) + [cls_token (B, 1, C)] -> (B, [1] + L, C)
            x ~ (B, C, T H, W) -> PatchEmbed(x) ~ (B, L, C) -> (B, L, C) + [cls_token (B, 1, C)] -> (B, [1] + L, C)
        :param x: (B, C, H, W) or (B, C, T, H, W)
        :param mask_dict: optional dict with keys 'mask' and 'ids_restore'
        :param use_checkpoint:
        :param return_features: if True, return features before the classification head
        :return: x (B, [1]+L, C) ([] means optional)
        """
        outs = dict()
        # x ~ (B, C, H, W) -> PatchEmbed(x) ~ (B, L, C) -> (B, L, C) + [cls_token (B, 1, C)] -> (B, L+[1], C),
        x, pos_embed = self.tokenize(x, mask_dict)

        # encoder attention blocks
        for blk in self.blocks:
            x = blk(x)

        # x ~ (B, [1]+L, C)
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

        elif self.classifier_feature == "cls_token" and not self.masked_image_modeling and not return_features:
            # case where we perform classification using only the cls_token
            assert self.first_patch_idx == 1, "Must have a CLS token at index 0"
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
                                      f"mim: {self.masked_image_modeling},"
                                      f"return_features: {return_features}"
                                      f"mask_dict: {mask_dict}")

    def forward(self, x: torch.Tensor,
                mask: Union[Dict[str, torch.Tensor], None] = None,
                return_features=False) -> Dict[str, torch.Tensor]:

        if self.classifier_feature is None:
            return_features = True
        elif not self.masked_image_modeling:
            assert return_features or self.classifier_feature in self.valid_classifier_features,\
                "If masked_image_modeling=False must have" \
                f" return_features=True, or classifier_feature='global_pool'" \
                f" instead got self.classifier_feature = {self.classifier_feature}," \
                f" return_features={return_features}"

        else:
            assert mask is not None, f"if masked_image_modeling=True must provide a mask"

        outs = self.forward_features(x, mask_dict=mask, return_features=return_features)
        return outs


class Decoder(VisionTransformerBase, ABC):
    valid_batch_masking_modes = ['symmetric', 'asymmetric']

    def __init__(
        self,
        attn_target: partial,
        patches_layout: List[int],
        first_patch_idx: int = 1,
        input_embed_dim: int = 768,
        embed_dim: int = 512,
        masked_image_modeling: bool = True,
        mask_token_embed_dim: Union[None, int] = None,
        decoder_depth: int = 8,
        mlp_ratio: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        share_pos_embed: bool = False,
        use_modality_token: bool = False,
        modality_token_mode: str = "add",
        modalities: Union[None, List] = None,
        use_all_seq: bool = False,
        batch_masking: str = 'symmetric'  # 'symmetric' or 'asymmetric
    ):

        #############################################  basic settings #############################################
        self.masked_image_modeling = masked_image_modeling
        self.patches_layout = patches_layout
        self.has_temporal_dim = patches_layout[0] > 1  # if True then pos_embed will be temporal and spatial
        num_patches = int(np.prod(patches_layout))
        use_cls_token = False  # we don't need to redefine a cls_token for the decoder
        self.share_pos_embed = share_pos_embed  # if sharing pos_embed with encoder then do not build it
        build_decoder_pos_embed = not share_pos_embed
        self.modalities = modalities
        self.batch_masking = batch_masking
        assert self.batch_masking in self.valid_batch_masking_modes, \
            f"batch_masking must be in {self.valid_batch_masking_modes} but got {self.batch_masking}"
        self.num_patches_per_modality = num_patches  # assume that each modality has the same number of patches
        ################################# init mask_token, cls_token and pos_embed #####################################
        super().__init__(num_patches, embed_dim, use_cls_token, mask_token_embed_dim, build_decoder_pos_embed,
                         use_modality_token=use_modality_token, modalities=modalities,
                         modality_token_mode=modality_token_mode, use_all_seq=use_all_seq)

        if self.use_all_seq:
            assert getattr(self, "insert_masked_tokens_seq") is not None, "insert_masked_tokens_seq must be defined " \
                                                                          "if use_all_seq=True"

        #############################################  attention blocks ################################################
        self.first_patch_idx = first_patch_idx  # override self.first_patch_idx determined above
        self.decoder_embed = nn.Linear(input_embed_dim, embed_dim, bias=True)
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]  # stochastic depth decay rule
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=None,
                    layer_scale_init_value=1e-4,
                )
                for i in range(decoder_depth)
            ]
        )

        self.norm = norm_layer(embed_dim)  # norm after the last block

        #############################################  init weights ################################################
        self.apply(self._init_weights)

    def __repr__(self):
        if self.modalities is not None:
            m = ''
            for mod in self.modalities:
                m += f'_{mod}'

            return f"{self.__class__.__name__}_{m}"

        else:
            return f"{self.__class__.__name__}"

    def forward(self, x, mask_dict: Union[Dict, None], modality=None):
        """ steps:
        decoder_embed(x) -> insert_masked_tokens -> decoder_blocks -> norm

        :param x: encoder output
        :param mask_dict: mask_dict['mask'] if given None then assumed that masked tokens are already inserted
        in x and visible tokens have been projected
        :param modality: modality only used for MultiModal* models
        :return: decoder output
        """

        pos_embed = self.pos_embed
        if mask_dict is not None:
            x = self.decoder_embed(x)
        if self.use_all_seq and mask_dict is not None:
            assert mask_dict.get('num_masked_tokens_per_modality') is not None, \
                f"if use_all_seq=True must provide num_masked_tokens_per_modality in mask_dict, " \
                f"got mask_dict with keys ={list(mask_dict.keys())}"
            # num_masked_tokens_per_modality=mask_dict['num_masked_tokens_per_modality'])
            x = self.insert_masked_tokens_seq(x, mask_dict)
            pos_embed = pos_embed.repeat(1, len(self.modalities), 1)  # repeat along the token dim
            # assumes that num_patches_per_modality are equal
        elif mask_dict is not None:
            x = self.insert_masked_tokens(x, mask_dict['mask'], modality=modality)
        # else: #assume masked tokens are already inserted in x externally
        #     a = 1 # todo: move add code from MAE_proto wiht self.insert_masked_tokens_asymetric

        x[:, self.first_patch_idx:] += pos_embed  # add pos_embed skipping cls_token and/or modality_token ('concat')

        if self.use_modality_token:  # fixme assume: modality_token_mode = 'add' when use_all_seq=True
            if self.use_all_seq:
                start = self.first_patch_idx
                for m in self.modalities:
                    end = start + self.num_patches_per_modality
                    x[:, start:end] += getattr(self, f"{m}_token")
                    start = end
            else:
                assert modality is not None, "modality must be provided if use_modality_token=True & use_all_seq False"
                if self.modality_token_mode == 'add':
                    x += getattr(self, f"{modality}_token")  # modality token of shape (1, 1, embed_dim)
                else:
                    # B = x.shape[0]
                    # add decoder modality token on top of modality token already in the seq
                    x[:, self.first_patch_idx - 1, :] += getattr(self, f"{modality}_token").squeeze(1)
                    # x = torch.cat([getattr(self, f"{modality}_token").repeat(B, 1, 1), x], dim=1)
                    # getattr(self, f"{modality}_token")

                    # x = torch.cat(, x], dim=1)

        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x)
        x = self.norm(x)
        return x


class DecoderCrossAttention(VisionTransformerBase, ABC):
    def __init__(
        self,
        attn_target: partial,
        cross_attn_target: partial,
        patches_layout: List[int],
        first_patch_idx: int = 1,
        input_embed_dim: int = 768,
        embed_dim: int = 512,
        masked_image_modeling: bool = True,
        mask_token_embed_dim: Union[None, int] = None,
        decoder_depth: int = 4,
        mlp_ratio: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_norm_eps: float = 1e-6,
        share_pos_embed=False,
        use_modality_token: bool = False,
        modality_token_mode: str = "add",
        modalities: Union[None, List] = None
    ):
        #############################################  basic settings #############################################
        self.masked_image_modeling = masked_image_modeling
        self.patches_layout = patches_layout
        self.has_temporal_dim = patches_layout[0] > 1  # if True then pos_embed will be temporal and spatial
        num_patches = int(np.prod(patches_layout))
        use_cls_token = False  # we don't need to redefine a cls_token for the decoder
        self.share_pos_embed = share_pos_embed  # if sharing pos_embed with encoder then do not build it
        build_decoder_pos_embed = not share_pos_embed
        self.modalities = modalities
        self.num_patches_per_modality = num_patches
        ################################# init mask_token, cls_token and pos_embed #####################################
        super().__init__(num_patches, embed_dim, use_cls_token, mask_token_embed_dim, build_decoder_pos_embed,
                         use_modality_token=use_modality_token, modalities=modalities,
                         modality_token_mode=modality_token_mode, use_all_seq=False)
        #############################################  attention blocks ################################################
        self.first_patch_idx = first_patch_idx  # override self.first_patch_idx determined above
        self.decoder_embed = nn.Linear(input_embed_dim, embed_dim, bias=True)
        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]

        # todo add option to BlockCrossAttention for 1st block and the rest are BlockAttention
        self.decoder_blocks = nn.ModuleList(
            [
                BlockCrossAttention(
                    dim=embed_dim,
                    attn_target=attn_target,
                    cross_attn_target=cross_attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=None,
                    layer_scale_init_value=1e-4,
                )
                for i in range(decoder_depth)
            ]
        )

        self.norm = norm_layer(embed_dim)  # norm after the last block

        #############################################  init weights ################################################
        self.apply(self._init_weights)

    def __repr__(self):
        if self.modalities is not None:
            m = ''
            for mod in self.modalities:
                m += f'_{mod}'

            return f"{self.__class__.__name__}_{m}"

        else:
            return f"{self.__class__.__name__}"

    def forward(self, x, mask_dict: Dict, context: Tensor, modality=None,
                context_embeddings: Union[Tensor, None] = None):
        pos_embed = self.pos_embed
        x = self.decoder_embed(x)
        context = self.decoder_embed(context)
        if self.use_modality_token:
            if self.modality_token_mode == 'add':
                # add modality token (externally provided from other decoder) to the context tokens
                assert context_embeddings is not None, f"context_embeddings must be provided if use_modality_token=True" \
                                                       f" instead got {context_embeddings}"
                context += context_embeddings
                # add modality token to the input tokens
                x += getattr(self, f"{modality}_token")  # modality token of shape (1, 1, embed_dim)
            else:
                raise NotImplementedError(f"modality_token_mode={self.modality_token_mode} not implemented")

        x = self.insert_masked_tokens(x, mask_dict['mask'], modality=modality)
        x[:, self.first_patch_idx:] += pos_embed  # add pos_embed skipping cls_token
        # todo add modality_token here
        for i, blk in enumerate(self.decoder_blocks):
            x = blk(x, context)
        x = self.norm(x)
        return x
