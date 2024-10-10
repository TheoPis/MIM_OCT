#!/usr/bin/env python3
import torch
import torch.nn as nn
from utils import DATASETS_INFO, printlog
from timm.models.layers import trunc_normal_
from typing import Union, List, Tuple
import math
import torch.nn.functional as F
from einops import rearrange
from functools import partial


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


class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net: [nn.Module],
        in_feature: str,
        out_channels: int,
        scale_factors: Union[List, Tuple],
        top_block=None,
        last_feat_channels: int = 1024,
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
        _out_feature_channels = {'last_feat': last_feat_channels}
        _out_feature_strides = {'last_feat': 14}
        # strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        self.strides = [4, 7, 14, 28]
        # _assert_strides_are_log2_contiguous(strides)
        printlog(f"SimpleFeaturePyramid strides: {self.strides}")
        # dim = input_shapes[in_feature].channels
        dim = _out_feature_channels[in_feature]
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

            # stage = int(math.log2(self.strides[idx]))
            stage = idx
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        # self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in self.strides}
        self._out_feature_strides = {"p{}".format(int(s)): s for s in self.strides}
        a = 1
        # top block output feature maps.
        # if self.top_block is not None:
        #     for s in range(stage, stage + self.top_block.num_levels):
        #         self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        # self._out_feature_channels = {k: out_channels for k in self._out_features}
        # self._size_divisibility = self.strides[-1]

    @property
    def patch_size(self):
        return self.net.patch_embed.patch_size[0]

    @property
    def embed_dim(self):
        return self.net.patch_embed.embed_dim

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
        # always get the last features here
        b, _, h, w = x.shape
        last_feats = self.net.get_intermediate_layers(x, n=1)[0]
        # feature spatial dims
        h = h // self.patch_size
        w = w // self.patch_size
        results = []
        last_feats = last_feats.reshape(b, h, w, self.embed_dim)
        last_feats = rearrange(last_feats, 'b h w c -> b c h w')  # reshape to shape for convs
        for stage in self.stages:
            results.append(stage(last_feats))
        # assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}


class DINOv2(nn.Module):
    eligible_backbones = ["dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"]
    eligible_phases = ['finetuning', 'linear_probing', 'fpn_probing']

    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.dataset = config['dataset']
        self.task = config.get('task', 'detection')
        self.backbone_name = config.get('backbone')
        self.phase = config.get('phase', 'finetuning')
        self.out_stride = config.get('out_stride', 16)
        self.align_corners = config.get('align_corners', False)
        self.num_classes = DATASETS_INFO[self.dataset].NUM_CLASSES[experiment]

        assert(self.backbone_name in self.eligible_backbones), \
            f'backbone_name {self.backbone_name} not in {self.eligible_backbones}'
        assert(self.phase in self.eligible_phases), 'mode must be in {}'.format(self.eligible_phases)

        self.train_head_only = False

        self.modalities = config.get('modalities', ['OCT'])

        patch_size = 14
        img_channels = 3

        self.use_all_seq = False  # needed with multimodal model that uses a single sequence as input
        self._get_backbone()
        if self.phase in ['fpn_probing']:
            self._get_segmentation_head()
        else:
            self.segmentation_head = None
        if self.phase == 'fpn_probing':
            self.freeze_backbone()

    def _get_backbone(self):
        if self.config['task'] == 'segmentation':
            if self.phase == 'finetuning':
                raise NotImplementedError()

            elif self.phase == 'fpn_probing':
                self.backbone = torch.hub.load('facebookresearch/dinov2', self.backbone_name, pretrained=True)

                self.fpn = SimpleFeaturePyramid(net=self.backbone,
                                                in_feature='last_feat',
                                                out_channels=256,
                                                scale_factors=(4, 2.0, 1.0, 0.5),
                                                top_block=None,
                                                last_feat_channels=1024 if 'dinov2_vitl14'
                                                                           in self.backbone_name else 1536,
                                                norm="LN"
                                                )

                # self.fpn.cuda()  # fixme not sure why this is not done already for fpn

            else:
                raise NotImplementedError(f"{self.phase} not implemented for segmentation task")
        else:
            if self.backbone_name in self.eligible_backbones:
                if self.phase == 'finetuning':
                    self.backbone = torch.hub.load('facebookresearch/dinov2', self.backbone_name, pretrained=True)
                    self.head = nn.Linear(self.backbone.embed_dim, self.num_classes)
                elif self.phase == 'linear_probing':
                    self.backbone = torch.hub.load('facebookresearch/dinov2', self.backbone_name, pretrained=True)
                    self.head = nn.Linear(self.backbone.embed_dim, self.num_classes)
                else:
                    raise NotImplementedError(f"{self.phase} not implemented for backbone {self.backbone_name}")

            else:
                raise NotImplementedError()

            if self.phase == 'linear_probing':
                self.get_linear_probing(self.config.get("use_bn_with_linear_probing", True))
            elif self.phase == 'finetuning':
                self.head = nn.Linear(self.backbone.embed_dim, self.num_classes)

    def freeze_backbone(self):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal """
        # freeze all but the head
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Freezing backbone --> trainable parameters: {n_trainable_parameters} "
                 f"-- frozen parameters: {n_frozen_parameters}")
        self.train_fpn_only = True


    def get_linear_probing(self, use_bn):
        """ Sets the backbone to frozen. Head is initialized with trunc_normal """
        # freeze all but the head
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False

        # re-init head
        printlog(f"re-init head with trunc_normal")
        trunc_normal_(self.head.weight, std=0.01)
        self.train_head_only = True
        if use_bn:
            self.head = nn.Sequential(
                torch.nn.Sequential(torch.nn.BatchNorm1d(self.backbone.embed_dim,
                                                         affine=False,
                                                         eps=1e-6),
                                    self.head
                                    )
            )
            printlog("Using BN in linear probing head")

        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Going to use linear probing --> trainable parameters: {n_trainable_parameters} "
                 f"-- frozen parameters: {n_frozen_parameters}")

    def _get_segmentation_head(self):
        fused_channels = len(self.fpn.strides) * self.fpn.out_channels  # 1024
        self.segmentation_head = nn.Sequential(nn.Conv2d(fused_channels,
                                                         self.num_classes,
                                                         kernel_size=1,
                                                         stride=1,
                                                         padding=0,
                                                         bias=True)
                                               )

    def fuse_features(self, feats: dict, input_size):
        fusion_list = []
        out_stride = min(self.fpn.strides)
        out_size = (input_size[0] // out_stride, input_size[1] // out_stride)
        interp = partial(nn.functional.interpolate, size=out_size, mode='bilinear', align_corners=self.align_corners)
        for key, feat in feats.items():
            # if self.fpn._out_feature_strides[key] == out_stride:
            #     fusion_list.append(feat)
            # else:
            fusion_list.append(interp(feat))
        fused_feats = torch.cat(fusion_list, 1)
        return fused_feats

    def forward(self, x):
        H, W = x.shape[-2:]
        # standard training (typically for supervised finetuning)
        if self.task == 'segmentation':
            feats_dict = self.fpn(x, modality=self.modalities[0])
            fused_feats = self.fuse_features(feats_dict, (H, W))
            out = self.segmentation_head(fused_feats)
            out = nn.functional.interpolate(out, size=(H, W), mode='bilinear', align_corners=self.align_corners)
        else:
            feats = self.backbone(x)
            out = self.head(feats)
        return out


