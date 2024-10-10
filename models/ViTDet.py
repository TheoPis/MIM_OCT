#!/usr/bin/env python3
import math
import os
import pathlib
from typing import Union, List, Tuple
from functools import partial
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import printlog, DATASETS_INFO, check_module_prefix, get_rank, is_distributed

from models.ViT_config import trunc_normal_
from models.mae.MAE_proto import MAE
from models.vit.ViT_layers import PadIm2Video, get_sinusoid_encoding_table, PatchEmbed

from torch.nn.modules.utils import _ntuple

to_2tuple = _ntuple(2)
__all__ = ["ViT", "SimpleFeaturePyramid", "VitDetMae"]

BACKBONES = \
    {"vit_tiny":
        {
            "patch_size": 16,
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3
        },
     "vit_small":
        {
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6
        },
     "vit_base":
        {
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "window_attention_blocks":
            [
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10
            ]
        },
     "vit_large":
         {
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "window_attention_blocks":
                [
                     # 6, 5, 8 11 for global attention
                     0, 1, 2, 3, 4,
                     6, 7, 8, 9, 10,
                     12, 13, 14, 15, 16,
                     18, 19, 20, 21, 22,
                ]
         }}


class DecoderLinear(nn.Module):
    def __init__(self, n_cls: int, patch_size: int, in_channels: int, img_size: Tuple[int, int]):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_cls = n_cls
        self.linear_layer = nn.Conv2d(in_channels=in_channels, out_channels=n_cls, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # H, W = self.img_size
        # GS = H // self.patch_size
        x = self.linear_layer(x)
        # x = rearrange(x, "b (h w) c -> b c h w", h=GS)
        return x


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
        out_channels (int): number of channels of the input
    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            # "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            # "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            # "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            # "nnSyncBN": nn.SyncBatchNorm,
            # "naiveSyncBN": NaiveSyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            # "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": lambda channels: LayerNorm(channels),  # fixme: using detectron2's LayerNorm implementation
        }[norm]
    return norm(out_channels)


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).
    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).
    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer: Union[nn.LayerNorm, partial] = nn.LayerNorm,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size: tuple=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # if self.use_residual_block:
        #     x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x


class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in from the paper
    "Exploring Plain Vision Transformer Backbones for Object Detection", https://arxiv.org/abs/2203.16527
    Its differences from models.ViT.VisionTransformer class are the following:
        1. It supports window attention.
        2. It supports relative positional embeddings.
        3. It supports output feature selection to be wrap-able around SimpleFeaturePyramid.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_cls_token=True,
        out_feature="last_feat",
        patch_embed_params_list: Union[List, None] = None
    ):
        super().__init__()
        assert len(img_size) == 2, f"input image size must be tuple/list of two integers instead got {img_size} "
        self.out_channels = embed_dim
        self.embed_dim = embed_dim # num_features for consistency with other models
        self.has_temporal_dim = False  # note: we never use temporal dimension with ViTDet backbone
        self.masked_image_modeling = False  # note: we never use patch_dropping with ViTDet backbone

        if patch_embed_params_list is None:
            patch_embed_params_list = [None]
        elif isinstance(patch_embed_params_list, list):
            if len(patch_embed_params_list) == 0:
                patch_embed_params_list = [None]
        else:
            raise ValueError(f"patch_embed_params_list must be a list or None instead got {patch_embed_params_list}")

        # patch_embed
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            pad_func=patch_embed_params_list[0]
        )
        num_patches = self.patch_embed.num_patches

        # cls_token
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            # self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.first_patch_idx = 1
            total_num_patches = num_patches + 1
        else:
            self.cls_token = None
            # self.pos_embed_cls = None
            self.first_patch_idx = 0
            total_num_patches = num_patches

        assert (self.patch_embed.patches_layout[-1] == self.patch_embed.patches_layout[-2]),\
            "Interpolation of pos embed not supported for non-square layouts"

        # pos_embed
        if use_abs_pos:
            self.register_buffer("pos_embed", get_sinusoid_encoding_table(num_patches, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(depth)]
        else:
            raise NotImplementedError(f"Drop path types are: [progressive, uniform]. Got {drop_path_type}.")

        # attention blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(self.patch_embed.num_patches, self.patch_embed.num_patches),
            )
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size[1]}
        # defaults to "last_feature" but when wrapped around SimpleFeaturePyramid itbd is modified to 4 layer_names
        self._out_features = [out_feature]
        self.apply(self._init_weights)

    @property
    def patch_size(self):
        return self.patch_embed.patch_size

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
            # todo impement interpolation for spatio-temporal pos_embed
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
        D = pos_embed.shape[-1]
        # unflatten pos_embed
        pos_embed = pos_embed.reshape(1, int(math.sqrt(L)), int(math.sqrt(L)), D)
        # pos_embed is (1, H//p, W//p, D) where H//p = W//p = sqrt(L)
        pos_embed = nn.functional.interpolate(
            pos_embed.permute(0, 3, 1, 2),  # (1, D, H//p, W//p)
            scale_factor=math.sqrt(target_L / L),
            mode="bicubic"
        )
        # pos_embed is (1, D, s * H//p, s * W//p) where s = sqrt(L' / L)
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, D)  # flatten to (1, L', D)
        return pos_embed

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

    def tokenize(self, x, mask_dict: Union[dict, None] = None):
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

        x = self.patch_embed(x)  # x ~ (B, L, D)
        num_tokens_per_img = x.shape[1]  # L (without cls_token)

        if self.cls_token is not None:  # modality token is the first token of the sequence by convention
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
                npatch_per_img=num_tokens_per_img,
                pos_embed=pos_embed,
                patches_layout=self.patch_embed.patches_layout,
                input_shape=input_shape,
                first_patch_idx=self.first_patch_idx,
                has_temporal_dim=self.has_temporal_dim
            )

        else:

            pos_embed = self.get_pos_embedding(
                npatch_per_img=num_tokens_per_img,
                pos_embed=self.pos_embed,
                patches_layout=self.patch_embed.patches_layout,
                has_temporal_dim=self.has_temporal_dim)

        # if self.add_pos_same_dtype:
        #     pos_embed = pos_embed.type_as(x)

        # add positional embedding to each token
        # if self.cls_token is not None:
        #     # if using cls_token, append a learnable token pos_embed_cls to pos_embed
        #     pos_embed = torch.cat([self.pos_embed_cls, pos_embed], dim=1)  # (1, L+1, C)

        x[:, self.first_patch_idx:] += pos_embed
        if self.masked_image_modeling and mask_dict is not None:
            x = self.remove_masked_tokens(x, mask_dict['mask'])
        # x = self.pos_drop(x)
        return x, pos_embed

    def forward_features(self, x, **kwargs):
        x, _ = self.tokenize(x)  # x ~ (B, L, D), pos_embed ~ (1, L, D) (unused for ViTDet after the tokenization)
        # ViTDet blocks expect input to be (B,h,w,C)
        h, w = self.patch_embed.patches_layout[1:]
        x = x[:, self.first_patch_idx:, :]  # remove cls_token  (if present)
        x = x.reshape(x.shape[0], h, w, x.shape[-1])  # (B, L, D) -> (B, h, w, D)
        for blk in self.blocks:
            x = blk(x)
        # fixme: from detectron: why B H W C -> B C H W
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs

    def forward(self, x, **kwargs) -> dict:
        outputs = self.forward_features(x, **kwargs)
        return outputs


class MultiViT(ViT):
    """
    Almost identical functionality to ViT, but with support for multiple modalities
    - modality-specific embedders
    -
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        drop_path_type="progressive",
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_cls_token=True,
        out_feature="last_feat",
        modality='OCT',
        use_modality_token=True,
    ):

        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, mlp_ratio, qkv_bias,
                         drop_path_rate, drop_path_type, norm_layer, act_layer, use_abs_pos, use_rel_pos,
                         rel_pos_zero_init, window_size, window_block_indexes, residual_block_indexes,
                         use_cls_token, out_feature)

        # overrides constructor of ViT and replaces (nn.Module) self.patch_embed with a ModuleDict
        printlog("Initializing MultiViT: replacing patch_embed with ModuleDict")
        del self.patch_embed
        self.patch_embeders = nn.ModuleDict()
        self.patch_embeders[modality] = PatchEmbed(img_size=img_size,
                                                   patch_size=patch_size,
                                                   in_chans=in_chans,
                                                   embed_dim=embed_dim)
        # self.modality_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.use_modality_token = use_modality_token
        self.modality_token_mode = 'add'  # 'add' or 'concat'

        if use_modality_token:
            modality_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            self.register_parameter(f"{modality}_token", modality_token)
            pos_embed_modality = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            self.register_parameter(f"pos_embed_{modality}", pos_embed_modality)
            # self.first_patch_idx += 1  # todo this is legacy, default now: modality_token is added not appended
        self.apply(self._init_weights)

    def tokenize(self, x, modality='OCT'):
        # todo
        # raise NotImplementedError("MultiViT.tokenize() must mirror new ViT.tokenize()")
        B = x.shape[0]
        x = self.patch_embeders[modality](x)  # x ~ (B, L, D)
        num_tokens_per_img = x.shape[1]  # L (without cls_token)
        seq = []

        # x becomes [cls_token, modality_token, x] (optionally)
        # if self.use_modality_token:
        #     modality_token = getattr(self, f'{modality}_token').expand(B, -1, -1)
        #     x = torch.cat((modality_token, x), dim=1)

        if self.cls_token is not None:  # modality token is the first token of the sequence by convention
            class_tokens = self.cls_token.expand(B, -1, -1)
            # x = torch.cat((class_tokens, x), dim=1)
            seq = [class_tokens] + seq

        # resize pos_embed in case of different sequence length
        pos_embed = self.get_pos_embedding(
            npatch_per_img=num_tokens_per_img,
            pos_embed=self.pos_embed,
            patches_layout=self.patch_embeders[modality].patches_layout,
            has_temporal_dim=self.has_temporal_dim,
            first_patch_idx=self.first_patch_idx
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

        seq = torch.cat(seq, dim=1)
        return seq, pos_embed

    def forward_features(self, x, **kwargs):
        modality = kwargs['modality']
        x, _ = self.tokenize(x, modality)
        h, w = self.patch_embeders[modality].patches_layout[1:]
        x = x[:, self.first_patch_idx:, :]  # remove cls_token and modality_token (if present)
        x = x.reshape(x.shape[0], h, w, x.shape[-1])  # (B, L, D) -> (B, h, w, D)
        for blk in self.blocks:
            x = blk(x)
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs

    def forward(self, x, **kwargs) -> dict:
        outputs = self.forward_features(x, **kwargs)
        return outputs


class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net: Union[ViT, MultiViT, nn.Module],
        in_feature: str,
        out_channels: int,
        scale_factors: Union[List, Tuple],
        top_block=None,
        norm="LN"
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (Tuple or list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further down-samples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
        """
        super(SimpleFeaturePyramid, self).__init__()
        # assert isinstance(net, Backbone)
        self.out_channels = out_channels  # 256 typically
        self.scale_factors = scale_factors

        # input_shapes = {in_feature: net._out_feature_channels[in_feature]}
        # net._out_feature_channels = {in_feature: embed_dim}
        # self._out_feature_strides = {out_feature: patch_size}
        #

        # strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        self.strides = [int(net._out_feature_strides[in_feature] / scale) for scale in scale_factors]
        # _assert_strides_are_log2_contiguous(strides)

        # dim = input_shapes[in_feature].channels
        dim = net._out_feature_channels[in_feature]
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            # todo merge conv2d with norm
            layers.extend([nn.Conv2d(out_dim,
                                     out_channels,
                                     kernel_size=1,
                                     bias=use_bias),
                           get_norm(norm, out_channels),
                           nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     padding=1,
                                     bias=use_bias),
                           get_norm(norm, out_channels)
                           ])

            layers = nn.Sequential(*layers)

            stage = int(math.log2(self.strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in self.strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = self.strides[-1]

    def forward(self, x, **kwargs):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
            kwargs: other inputs passed to the underlying backbone's forward.
                modality: str, e.g "OCT"
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.net(x, **kwargs)
        features = bottom_up_features[self.in_feature]
        # features = rearrange(features, 'b c h w -> b h w c')
        results = []

        for stage in self.stages:
            # features = rearrange(features, 'b c h w -> b h w c')
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


def vitd_finetune_single_modality(out_features: int,
                                  backbone: str,
                                  patch_size: int = 16,
                                  crop_size: int = 224,
                                  window_size: int = 16,
                                  in_channels=1,
                                  drop_path_rate=0.0,
                                  use_cls_token=False,
                                  use_modality_token=False,
                                  use_rel_pos=False,
                                  pretrained=True,
                                  internal_checkpoint_path: Union[str, None] = None,
                                  use_cls_token_pretraining=True,
                                  mm_pretrained=False,
                                  modality='OCT'):

    patch_size = [patch_size, patch_size] if type(patch_size) == int else patch_size
    crop_size = [crop_size, crop_size] if type(crop_size) == int else crop_size

    # assert type(patch_size) == int and type(crop_size) == int,\
    #     f"patch_size and crop_size must be int instead got types {type(patch_size)} and {type(crop_size)}"
    if len(patch_size) == 2:
        patch_embed_type = "linear"
        patch_embed_params = [None]
    elif len(patch_size) == 3:  # case of 3D (BCTHW) input
        patch_embed_type = "linear"
        if len(crop_size) == 2:
            patch_embed_params = [PadIm2Video(ntimes=patch_size[0], pad_type="repeat")]
            printlog(f"Given a 3d patch_size [{patch_size}] but crop_size is 2D [{crop_size}] "
                     f"so using PadIm2Vdeo to make input BCTHW")
        else:
            patch_embed_params = [None]

    else:
        raise ValueError(f"patch_size must be of length 2 or 3, got {len(patch_size)} instead {patch_size}")

    backbone_settings = BACKBONES[backbone]
    embed_dim = backbone_settings["embed_dim"]
    depth = backbone_settings["depth"]
    num_heads = backbone_settings["num_heads"]
    window_attention_blocks = backbone_settings["window_attention_blocks"] if window_size > 0 else []
    if not (window_size > 0):
        printlog(f"window_size set to {window_size}: all layers use global self-attention ! ! ")
    printlog(f"building {backbone} (multimodal: {mm_pretrained}, modality: {modality}) for finetuning with\n"
             f"window_attention_blocks: {window_attention_blocks}\n"
             f"window_size: {window_size}\n"
             f"patch_size: {patch_size}\n"
             f"crop_size: {crop_size}\n"
             f"out_features:{out_features}\n"
             f"in_channels: {in_channels}\n"
             f"embed_dim: {embed_dim}\n"
             f"depth:{depth}\n"
             f"num_heads: {num_heads}\n"
             f"drop_path_rate: {drop_path_rate}\n"
             f"use_cls_token: {use_cls_token}\n"
             f"use_rel_pos: {use_rel_pos}\n"
             f"pretrained: {pretrained}\n"
             f"internal_checkpoint_path: {internal_checkpoint_path}\n"
             f"use_cls_token_pretraining: {use_cls_token_pretraining}")

    # patch_size = [patch_size, patch_size] if type(patch_size) == int else patch_size
    # crop_size = [crop_size, crop_size] if type(crop_size) == int else crop_size
    # printlog(f"initializing from internal pretrained checkpoint: {internal_checkpoint_path}")
    # todo modify ViT class to accept multimodal embedder a la VisionTransformer from ViT.py
    if mm_pretrained:
        enc = MultiViT(
            use_cls_token=use_cls_token,
            use_modality_token=use_modality_token,
            img_size=crop_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_chans=in_channels,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            window_size=window_size,
            mlp_ratio=4,  # fixme this always 4 for all vit models
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=window_attention_blocks,
            residual_block_indexes=[],
            use_rel_pos=use_rel_pos,
            out_feature="last_feat",
            modality=modality)

    else:
        enc = ViT(
            use_cls_token=use_cls_token,
            img_size=crop_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            in_chans=in_channels,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
            window_size=window_size,
            mlp_ratio=4,  # fixme this always 4 for all vit models
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=window_attention_blocks,
            residual_block_indexes=[],
            use_rel_pos=use_rel_pos,
            out_feature="last_feat",
            patch_embed_params_list=patch_embed_params
        )

    head = torch.nn.Identity
    model = MAE(enc, None, head)
    return model


class VitDetMae(nn.Module):
    eligible_backbones = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']
    eligible_phases = ['pretraining', 'finetuning', 'scratch', 'linear_probing', 'fpn_probing']

    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.dataset = config['dataset']
        self.task = config['task']
        self.backbone_name = config.get('backbone')
        self.phase = config.get('phase')
        self.out_stride = config.get('out_stride', 16)
        self.norm = config.get('norm', None)
        self.num_classes = DATASETS_INFO[self.dataset].NUM_CLASSES[experiment]

        assert(self.backbone_name in self.eligible_backbones), \
            f'backbone_name {self.backbone_name} not in {self.eligible_backbones}'
        assert(self.phase in self.eligible_phases), 'mode must be in {}'.format(self.eligible_phases)

        self.align_corners = config['align_corners'] if 'align_corners' in config else False  # fixme unused
        self.dropout = config['dropout'] if 'dropout' in config else 0.0  # fixme unused

        self.backbone_settings = config.get('backbone_settings', {})
        self.backbone_pretrained = config.get('pretrained', False)

        self.train_head_only = self.phase == 'linear_probing'
        self.train_fpn_only = self.phase == 'fpn_probing'

        self.modalities = config.get('modalities', ['OCT'])
        self.get_intermediate = True
        self.return_all_scales = True if 'ms_projector' in config else False

        # fixme: this is a bit of a mess, but it works for now, the main issue is that
        # fixme: external checkpoints have slightly different param name (ex 'backbone.trunk.pos_embed' vs 'pos_embed')
        # fixme: use cls_token and in_channels = 3
        # fixme: whereas internal checkpoints have cls_token=False and in_channels = 1

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

        # todo this leads to error when in inference mode
        # if (not self.backbone_pretrained) \
        #         and (not self.internal_backbone_pretrained) \
        #         and (not self.external_backbone_pretrained)\
        #         and self.phase == 'finetuning':
        #     # if not checkpoint is given and we are in finetuning mode we assume that we are training from scratch
        #     self.phase = 'scratch'
        self._get_backbone()
        if self.phase in ['fpn_probing', 'finetuning']:
            self._get_fpn()
            self._get_segmentation_head()
        else:
            self.segmentation_head = None
        if self.phase == 'fpn_probing':
            self.freeze_backbone()

    def _get_backbone(self):
        self.modality = None
        # note: if config['graph']['pretrain_modalities'] with len>=2 then model was multimodal-y pretrained
        pretrain_modalities = self.config.get('pretrain_modalities', ['OCT'])
        if pretrain_modalities is None:
            pretrain_modalities = [None]
        self.mm_pretrained = len(pretrain_modalities) > 1
        printlog(f'Pretrained model is_multimodal: {self.mm_pretrained}'
                 f' with pretrain_modalities: {pretrain_modalities}')
        if self.mm_pretrained:
            printlog(f'Pretraining modalities were "{self.config["pretrain_modalities"]}"')
            finetune_modalities = self.config.get('modalities')
            if len(finetune_modalities) > 1:
                raise NotImplementedError('Multi-modality finetuning currently not implemented')
            else:
                self.modality = finetune_modalities[0]
                printlog(f'{self.phase} with single modality {self.modality}')

        if self.backbone_name in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'vit_huge']:
            if self.phase == 'pretraining':
                raise NotImplementedError(f'{self.backbone_name} with phase {self.phase} not implemented')

            elif self.phase == 'finetuning':
                if self.mm_pretrained and self.config.get('cm', False):
                    raise NotImplementedError('Multimodal model finetuning currently not implemented ')
                    # todo rewrite this
                    # self.backbone = mvitd_cm_finetune_single_modality(out_features=self.num_classes,
                    #                                                   backbone=self.backbone_name,
                    #                                                   **self.config['backbone_settings'],
                    #                                                   pretrained=self.backbone_pretrained,
                    #                                                   mm_pretrained=self.mm_pretrained,
                    #                                                   modality=self.modality)
                else:
                    self.backbone = vitd_finetune_single_modality(out_features=self.num_classes,
                                                                  backbone=self.backbone_name,
                                                                  **self.config['backbone_settings'],
                                                                  pretrained=self.backbone_pretrained,
                                                                  mm_pretrained=self.mm_pretrained,
                                                                  modality=self.modality)
            elif self.phase == 'scratch':
                self.backbone = vitd_finetune_single_modality(out_features=self.num_classes,
                                                              backbone=self.backbone_name,
                                                              **self.config['backbone_settings'],
                                                              pretrained=False,
                                                              mm_pretrained=self.mm_pretrained,
                                                              modality=self.modality)

            elif self.phase in ['linear_probing', 'fpn_probing']:
                self.backbone = vitd_finetune_single_modality(out_features=self.num_classes,
                                                              backbone=self.backbone_name,
                                                              **self.config['backbone_settings'],
                                                              pretrained=self.backbone_pretrained,
                                                              mm_pretrained=self.mm_pretrained,
                                                              modality=self.modality)

        else:
            raise NotImplementedError(f'{self.backbone_name} with phase {self.phase} not implemented')
        # todo: does not support spatio-temporal models
        if self.internal_checkpoint_path:
            self._load_internal_pretrained_backbone(remove_key_from_checkpoint=('head', 'mask_token', 'decoder', 'encoders_cm'))

        elif self.external_checkpoint_path:
            self._load_external_pretrained_backbone(remove_key_from_checkpoint=None)

        if self.phase == 'linear_probing':
            self.get_linear_probing(use_bn=self.config.get("use_bn_with_linear_probing", True))

    def _get_fpn(self):
        if hasattr(self.backbone, 'encoder'):
            self.fpn = SimpleFeaturePyramid(net=self.backbone.encoder,
                                            in_feature='last_feat',
                                            out_channels=256,
                                            scale_factors=(4.0, 2.0, 1.0, 0.5),
                                            top_block=None,
                                            norm="LN"
                                            )
        elif hasattr(self.backbone, 'encoders'): # todo we assume single_modality finetuning hence only
            self.fpn = SimpleFeaturePyramid(net=self.backbone.encoders[self.modality],
                                            in_feature='last_feat',
                                            out_channels=256,
                                            scale_factors=(4.0, 2.0, 1.0, 0.5),
                                            top_block=None,
                                            norm="LN"
                                            )
        else:
            raise ValueError("backbone that does not have trunk or encoders attribute is not supported")

    def _get_segmentation_head(self):
        fused_channels = len(self.fpn.strides) * self.fpn.out_channels  # 1024
        self.segmentation_head = nn.Sequential(nn.Conv2d(fused_channels,
                                                         self.num_classes,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0,
                                                         bias=True)
                                               )

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
                # assume checkpoint_state_dict['pos_embed'] always includes pos_embed_cls at index 0
                checkpoint_state_dict['pos_embed'] = pos_embed_checkpoint[:, 1:]
                if self.backbone.encoder.use_cls_token:
                    # split cls token from pos_embed in checkpoint
                    checkpoint_state_dict['pos_embed_cls'] = pos_embed_checkpoint[:, 0].unsqueeze(0)
        else:
            raise NotImplementedError(f"self.backbone should have either encoders or encoder attribute")

        return pos_current, checkpoint_state_dict, remove_key_from_checkpoint

    def _get_pos_embeddings_internal(self, checkpoint_state_dict, remove_key_from_checkpoint=None):

        if 'backbone.encoder.pos_embed' in checkpoint_state_dict:
            # standard case: load the pos_embed (which is typically non-learnable) from the checkpoint
            pos_embed_checkpoint = checkpoint_state_dict['backbone.encoder.pos_embed']
            pos_embed_cls_checkpoint = checkpoint_state_dict.get('backbone.encoder.pos_embed_cls', None)
            if pos_embed_cls_checkpoint is not None:
                printlog("WARNING !!! legacy checkpoint: pos_embed_cls found in checkpoint, WARNING !!!! ")
            pos_embed_current = self.backbone.encoder.pos_embed

        elif self.phase == 'finetuning' and 'backbone.encoder.pos_embed' not in self.state_dict():
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
        assert self.phase in ['finetuning', 'pretraining', 'linear_probing'],\
            f'init with external pretrained backbone only supported for phase in [finetuning, pretraining]' \
            f' instead got phase: {self.phase}'

        if 'vit' not in self.external_checkpoint_path and 'RETFound' not in self.external_checkpoint_path:
            raise NotImplementedError(f'external checkpoint {self.external_checkpoint_path} not supported')

        assert self.config['backbone_settings']['in_channels'] == 3, f'external checkpoint was trained on rgb images,'\
                                                                     f' please specify' \
                                                                     f' graph.backbone_settings.in_channels=3 in config'
        assert self.config['backbone_settings']['use_cls_token'], f'external checkpoint was trained with use_cls_token'\
                                                                  f' please specify' \
                                                                  f' graph.backbone_settings.use_cls_token=true ' \
                                                                  f'in config'

        # fixme only support vit_mae from official mae repo
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

        if remove_key_from_checkpoint is not None:
            checkpoint_state_dict = self.remove_keys_from_checkpoint(checkpoint_state_dict,
                                                                     remove_key_from_checkpoint)

        printlog(f"Loading from {self.external_checkpoint_path}")
        if hasattr(self.backbone, 'encoder'):
            # case of unimodal or multimodal model with shared encoder
            missing_keys_from_chkpt, missing_keys_from_model = \
                self.backbone.encoder.load_state_dict(checkpoint_state_dict, strict=False)
            printlog(f'{len(missing_keys_from_chkpt)} keys in model but NOT in chkpt: {missing_keys_from_chkpt}')
            printlog(f'{len(missing_keys_from_model)} keys in chkpt but NOT in model: {missing_keys_from_model}')
            printlog(f"************************************")
        else:
            raise ValueError(f'external backbone [{self.backbone}] mut have "encoder" attribute')

    def _load_internal_pretrained_backbone(self, remove_key_from_checkpoint: Union[str, tuple, None] = None):
        """ loads an internally pretrained backbone (e.x from ssl on KEKI)
        :param remove_key_from_checkpoint: when finetuning we typically do not want to load the some chkpt keys
                                           so remove keys that contain  from the checkpoint
        """
        assert self.phase in ['finetuning', 'pretraining', 'linear_probing', 'fpn_probing'],\
            f'init with internal pretrained backbone only supported for phase in [finetuning, pretraining]' \
            f' instead got phase: {self.phase}'

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
                printlog("single-modality finetuning of a multimodal checkpoint:"
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

            elif self.phase in ['finetuning', 'linear_probing', 'fpn_probing'] and len(self.modalities) == 1:
                modalities_in_checkpoint = self.config.get('pretrain_modalities', ['OCT'])
            else:
                raise ValueError(f'phase: {self.phase} and modalities: {self.modalities} not supported')

            # remove modalities that are not in checkpoint
            remove_key_from_checkpoint = list(remove_key_from_checkpoint)
            for modality in modalities_in_checkpoint:
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

    def freeze_backbone(self):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal """
        # freeze all but the head
        for _, p in self.backbone.encoder.named_parameters():
            p.requires_grad = False
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Freezing backbone --> trainable parameters: {n_trainable_parameters} "
                 f"-- frozen parameters: {n_frozen_parameters}")
        self.train_fpn_only = True

    def get_linear_probing(self, use_bn):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal """
        # freeze all but the head
        for _, p in self.backbone.encoder.named_parameters():
            p.requires_grad = False
        # for _, p in self.backbone.head.named_parameters():
        #     p.requires_grad = True

        # patch size and img size are irrelevant todo: remove !
        self.backbone.head = DecoderLinear(self.num_classes,
                                           self.backbone.patch_size,
                                           self.backbone.encoder.embed_dim,
                                           (self.backbone_settings['crop_size'],
                                            self.backbone_settings['crop_size']))

        # re-init head
        printlog(f"re-init head with trunc_normal")
        trunc_normal_(self.backbone.head.linear_layer.weight, std=0.01)
        self.train_head_only = True
        # if use_bn:
        #     self.backbone.head = nn.Sequential(
        #         torch.nn.Sequential(torch.nn.BatchNorm1d(self.backbone.head.in_features,
        #                                                  affine=False,
        #                                                  eps=1e-6),
        #                             self.backbone.head
        #                             )
        #     )
        #     printlog("Using BN in linear probing head")
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Going to use linear probing --> trainable parameters: {n_trainable_parameters} "
                 f"-- frozen parameters: {n_frozen_parameters}")
        # overwrite existing head
        printlog(f"overwriting existing head {self.backbone.head} with DecoderLinear ... ")

    def fuse_features(self, feats: dict, input_size):
        fusion_list = []
        out_stride = min(self.fpn.strides)
        out_size = (input_size[0] // out_stride, input_size[1] // out_stride)
        interp = partial(nn.functional.interpolate, size=out_size, mode='bilinear', align_corners=self.align_corners)
        for key, feat in feats.items():
            if self.fpn._out_feature_strides[key] == out_stride:
                fusion_list.append(feat)
            else:
                fusion_list.append(interp(feat))
        fused_feats = torch.cat(fusion_list, 1)
        return fused_feats

    def forward(self, x, is_training=True, **kwargs):
        H, W = x.shape[-2:]
        input_kwargs = {}
        if is_training and self.phase == "pretraining":
            raise NotImplementedError("Pretraining for ViTDet not implemented")
        else:

            # if self.is_multimodal:
            #     assert 'modality' in kwargs, f"modality must be specified in kwargs for multimodal model"
            #     modality = kwargs['modality']
            # else:
            #     modality = None  # has no effect
            modality = kwargs.get('modality', None)

            # classification or biomarker detection head
            if self.task == 'detection':
                raise NotImplementedError("Detection head not implemented yet")
            elif self.task == 'segmentation':
                if self.modality == 'IR':
                    input_kwargs.update({'modality': 'IR'})
                if self.phase == 'linear_probing':
                    feats = self.backbone.encoder(x, **input_kwargs)['last_feat']
                    out = self.backbone.head(feats)
                    out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=self.align_corners)

                else: # finetuning or fpn_probing

                    feats_dict = self.fpn(x, **input_kwargs)  # includes self.backbone.trunk(x)
                    feats = self.fuse_features(feats_dict, (H, W))
                    out = self.segmentation_head(feats)  # logits B, num_classes, H, W
                    out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=self.align_corners)

            else:
                raise NotImplementedError(f"task {self.task} not implemented yet")
        return out
