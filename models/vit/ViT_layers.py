#!/usr/bin/env python3
from functools import partial
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.modules.utils import _ntuple
to_2tuple = _ntuple(2)


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Sinusoid position encoding table (assuming input x is (B, L, D))
    :param n_position: L i.e number of patches
    :param d_hid: patch embedding dimension D
    :return: (1, L, D)
    """
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # plt.imshow(sinusoid_table)
    # plt.show()

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PadIm2Video(torch.nn.Module):
    def __init__(self, ntimes, pad_type, time_dim=2):
        super().__init__()
        self.time_dim = time_dim
        assert ntimes > 0
        assert pad_type in ["zero", "repeat"]
        self.ntimes = ntimes
        self.pad_type = pad_type

    def forward(self, x):
        if x.ndim == 4:
            # B, C, H, W -> B, C, T, H, W
            x = x.unsqueeze(self.time_dim)

        if x.shape[self.time_dim] == 1:
            if self.pad_type == "repeat":
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.repeat(new_shape)
            elif self.pad_type == "zero":
                padarg = [0, 0] * len(x.shape)
                padarg[2 * self.time_dim + 1] = self.ntimes - x.shape[self.time_dim]
                x = torch.nn.functional.pad(x, padarg)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version,
        # can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # B,N,3C -> B, N, 3, H, C//H -> 3, B, H, N, C//H
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        # multiheaded attenion: each head operates over C//H of each tokens dimensions
        # q is B, H, N, C//H
        # k is B, H, N, C//H --> k.transpose(-2, -1) is B, H, C//H, N
        # v is B, H, N, C//H
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn is B, H, N, N
        attn = attn.softmax(dim=-1)
        # attn is B, H, N, N
        attn = self.attn_drop(attn)
        # attn @ v gives B, H, N, C//H
        # transpose(1, 2) gives B, N, H, C//H
        # reshape gives B, N, C
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, N, C = x.shape
        _, M, _ = context.shape

        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        attn_target,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer: Union[nn.LayerNorm, partial] = nn.LayerNorm,
        layer_scale_type=None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,  # from cait; float
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if isinstance(attn_target, nn.Module):
            self.attn = attn_target
        else:
            self.attn = attn_target(dim=dim)

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale_type = layer_scale_type

        # Layerscale
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]
            # two gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )
            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.layer_scale_gamma1)
            x = x + self.drop_path(self.mlp(self.norm2(x)) * self.layer_scale_gamma2)
        return x

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr = (
                    string_repr
                    + "("
                    + name
                    + "): "
                    + "tensor("
                    + str(tuple(p[1].shape))
                    + ", requires_grad="
                    + str(p[1].requires_grad)
                    + ")\n"
                )

        return string_repr


class BlockCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        attn_target,
        cross_attn_target,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer: Union[nn.LayerNorm, partial] = nn.LayerNorm,
        layer_scale_type=None,  # from cait; possible values are None, "per_channel", "scalar"
        layer_scale_init_value=1e-4,  # from cait; float
    ):
        super().__init__()
        if isinstance(attn_target, nn.Module):
            self.attn = attn_target
        else:
            self.attn = attn_target(dim=dim)

        if isinstance(cross_attn_target, nn.Module):
            self.cross_attn = cross_attn_target
        else:
            self.cross_attn = cross_attn_target(dim=dim)

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm_context = norm_layer(dim)

        if drop_path > 0.0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale_type = layer_scale_type

        # Layerscale
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in ["per_channel", "scalar"], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                # one gamma value per channel
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                # single gamma value for all channels
                gamma_shape = [1, 1, 1]

            # three gammas: for each part of the fwd in the encoder
            self.layer_scale_gamma1 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

            self.layer_scale_gamma2 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

            self.layer_scale_gamma3 = nn.Parameter(
                torch.ones(size=gamma_shape) * layer_scale_init_value,
                requires_grad=True,
            )

    def forward(self, x, context):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.cross_attn(self.norm2(x), self.norm_context(context)))
            x = x + self.drop_path(self.mlp(self.norm3(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)) * self.layer_scale_gamma1)
            x = x + self.drop_path(self.cross_attn(self.norm2(x), context) * self.layer_scale_gamma2)
            x = x + self.drop_path(self.mlp(self.norm3(x)) * self.layer_scale_gamma3)
        return x

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ""
        for p in self.named_parameters():
            name = p[0].split(".")[0]
            if name not in named_modules:
                string_repr = (
                    string_repr
                    + "("
                    + name
                    + "): "
                    + "tensor("
                    + str(tuple(p[1].shape))
                    + ", requires_grad="
                    + str(p[1].requires_grad)
                    + ")\n"
                )

        return string_repr


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self,
                 img_size: Union[int, tuple, list] = 224,
                 patch_size=16,
                 in_chans: int = 3,
                 embed_dim: int = 768,
                 pad_func: Union[None, PadIm2Video] = None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.pad_func = pad_func
        if pad_func is not None:
            assert isinstance(pad_func, PadIm2Video), f"pad_func must be None or PadIm2Video, got {pad_func}"

        if len(img_size) == 3:
            # case where input is 3D and patch_size is 3D so Conv3d is used
            assert len(patch_size) == 3, f"3D data require 3D patches " \
                                         f"instead got img_size: {img_size} and patch_size {patch_size}"
            self.patches_layout = tuple([img_size[i] // patch_size[i] for i in range(3)])
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        elif len(img_size) == 2:
            if self.pad_func is None:
                # case where input is 2D and we don't pad so Conv2d is used
                assert len(patch_size) == 2, f"2D data require 2D patches " \
                                             f"instead got img_size: {img_size} and patch_size {patch_size}"
                self.patches_layout = tuple([img_size[i] // patch_size[i] for i in range(2)])
                self.patches_layout = (1, *self.patches_layout)  # for 2D data, we add a dummy dimension
                self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

            else:
                # case where input is 2D and we pad so Conv3d is used
                assert len(patch_size) == 3, f"2D data require 2D patches " \
                                             f"instead got img_size: {img_size} and patch_size {patch_size}"
                self.patches_layout = tuple([img_size[i] // patch_size[i+1] for i in range(2)])
                self.patches_layout = (1, *self.patches_layout)  # for 2D data, we add a dummy dimension
                self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        else:
            raise ValueError(f"img_size: {img_size} and patch_size {patch_size} "
                             f"are not valid for PatchEmbed -- 2 or 3 elements")

        self.num_patches = np.prod(self.patches_layout)

    def forward(self, x):
        if self.pad_func is not None:
            x = self.pad_func(x)  # note: identity or PadIm2Video(ntimes=2, repeat)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbedGeneric(nn.Module):
    """
    PatchEmbed that handles both images and videos (or 3D images)
    """

    def __init__(self, proj_stem, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        if len(proj_stem) > 1:
            self.proj = nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]
        # get the num_patches
        assert (
            isinstance(img_size, list) and len(img_size) >= 3
        ), "Need the full C[xT]xHxW in generic"
        # compute num_tokens with a forward
        with torch.no_grad():
            dummy_img = torch.zeros(
                [
                    1,
                ]
                + img_size
            )
            self.patches_layout = tuple(self.proj(dummy_img).shape[2:])
            self.num_patches = np.prod(self.patches_layout)

    def forward(self, x):
        # rgirdhar: no flatten here since the projection can handle it in the list of ops
        x = self.proj(x)
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)


class CrossModalTransformer(nn.Module):
    """wraps together a series of SA-CA-MLP blocks where CA uses as context embeddings from another modality"""
    def __init__(
        self,
        embed_dim=768,
        depth=12,
        mlp_ratio=4,
        attn_target=None,
        cross_attn_target=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_path_type="progressive",
        use_cls_token=True,
        layer_scale_type=None,
        layer_scale_init_value=1e-4,
        layer_norm_eps=1e-6,
        add_pos_same_dtype=False,
        masked_image_modeling=False,
        mask_token_embed_dim=None
    ):
        super().__init__()
        assert use_cls_token
        self.embed_dim = embed_dim
        self.add_pos_same_dtype = add_pos_same_dtype
        self.masked_image_modeling = masked_image_modeling

        if self.masked_image_modeling:
            # initialized to zeros following iBOT
            if mask_token_embed_dim is None:
                mask_token_embed_dim = embed_dim
            self.mask_token = nn.Parameter(torch.zeros(1, mask_token_embed_dim))

        norm_layer = partial(nn.LayerNorm, eps=layer_norm_eps)
        if use_cls_token:
            self.first_patch_idx = 1
        else:
            self.first_patch_idx = 0
        self.use_cls_token = use_cls_token

        # stochastic depth decay rule
        if drop_path_type == "progressive":
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(depth)]
        else:
            raise NotImplementedError(f"unknown drop_path_type {drop_path_type} expected 'progressive' or 'uniform'")

        self.blocks = nn.ModuleList(
            [
                BlockCrossAttention(
                    dim=embed_dim,
                    attn_target=attn_target,
                    cross_attn_target=cross_attn_target,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(depth)
            ]
        )

        # initialize
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def insert_masked_tokens(self, x, ids_restore):
        """ insert mask tokens x at positions ids_restore. Repeats mask_token to match batch_size and sequence_length
        follows https://github.com/facebookresearch/mae/blob/main/models_mae.py
        :param x: unmasked tokens (B, [1] + num_patches_kept, D)
        :param ids_restore: (B, [1] + L) contains the permutation (per batch element) that if applied to the sequence
         concat(tokens,masked_tokens) would restore them to their original order (i.e the image ordering)
        :return: x (B, [1] + L, D)
        """
        B, num_patches_kept, D = x.shape
        sequence_length = ids_restore.shape[1]
        if self.first_patch_idx > 0:
            num_patches_kept -= self.first_patch_idx  # first tokens are cls and/or modality tokens not : masked tokens
        num_patches_masked = sequence_length - num_patches_kept  # num_patches_masked = L - num_patches_kept
        # repeat mask_token num_patches_masked times
        mask_tokens = self.mask_token.repeat(B, num_patches_masked, 1)  # mask_tokens (B, num_patches_masked, D)
        x_ = torch.cat([x[:, self.first_patch_idx:, :], mask_tokens], dim=1)  # remove cls token --> x_ (B, L, D)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # restore ordering
        # append cls and/or modality token if used (else nothing happens)
        x = torch.cat([x[:, :self.first_patch_idx, :], x_], dim=1)
        return x

    def forward(self, x, context, mask: Union[Dict, None]=None):
        for blk in self.blocks:
            x = blk(x, context)
        if mask is not None:
            x = self.insert_masked_tokens(x, mask["ids_restore"])
        return x


class AttentionPoolingClassifier(nn.Module):
    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 1,
        use_linear: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim must be divisible by num_heads instead got dim: {dim} and heads: {num_heads}"
        if out_features != -1 and not use_linear:
            warnings.warn(f"In AttentionPoolingClassifier: use_linear: {use_linear}"
                          f" and out_features: {out_features}. No linear layer will be created ")
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        if use_linear:
            self.linear = nn.Linear(dim, out_features)
        else:
            self.linear = torch.nn.Identity()
        self.bn = nn.BatchNorm1d(dim, affine=False, eps=1e-6)

        self.num_queries = num_queries

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        B, N, C = x.shape

        x = self.bn(x.transpose(-2, -1)).transpose(-2, -1)
        cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(
            B, self.num_queries, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        q = q * self.scale
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, self.num_queries, C)
        x_cls = x_cls.mean(dim=1)

        out = self.linear(x_cls)
        return out


class AttentionPoolingClassifierMultimodal(nn.Module):
    valid_pooling_types = ["per_modality", "from_modality"]

    def __init__(
        self,
        dim: int,
        out_features: int,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 1,
        modalities: tuple = ('OCT', 'IR'),
        pooling_type: str = "per_modality",
    ):
        super().__init__()
        assert pooling_type in self.valid_pooling_types, f"pooling_type must be one of {self.valid_pooling_types} " \
                                                         f"instead got {pooling_type}"
        assert len(modalities) > 1, f"num_modalities must be greater than 1, got {len(modalities)}"
        self.modalities = modalities
        self.pooling_type = pooling_type
        self.attentive_pooling = nn.ModuleDict()
        for m in self.modalities:
            self.attentive_pooling[m] = AttentionPoolingClassifier(
                dim=dim,
                out_features=out_features,  # this does not affect anything as we remove the linear layer
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                num_queries=num_queries,
                use_linear=False
            )
        if self.pooling_type == 'per_modality':
            self.linear = nn.Linear(len(self.modalities)*dim, out_features)

    def forward(self, x: torch.Tensor, tokens_per_modality, **_: Any):
        out = []
        for m in self.modalities:
            out.append(self.attentive_pooling[m](x))
        out = self.linear(torch.concat(out, dim=-1))  # channel-wise concat
        return out
