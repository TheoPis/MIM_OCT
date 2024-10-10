#!/usr/bin/env python3
from functools import partial
from typing import Union, List
import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from utils import printlog

from models.vit.ViT import (
        Attention,
        CrossAttention,
        Decoder,
        DecoderCrossAttention,
        PadIm2Video,
        VisionTransformer,
        CrossModalTransformer
    )
from models.mvit.MViT import MultimodalVisionTransformer
from models.mvit.MViT_Seq import MultimodalVisionTransformerSeq
from models.mae.MAE_proto import MultiModalMAE, MAE, MultiModalMAESeq

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
            "num_heads": 12
        },
     "vit_large":
         {
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16
         },

     "vit_huge":
         {
            "patch_size": 16,
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16
         }
}


def make_conv_or_linear(layer, init_weight=None, init_bias=None):
    if init_weight is not None:
        init_weight(tensor=layer.weight.data)
    if init_bias is not None:
        init_bias(tensor=layer.bias.data)
    return layer


def reshape_and_init_as_mlp(tensor):
    # Based on MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
    torch.nn.init.xavier_uniform_(tensor.view([tensor.shape[0], -1]))


def vit_mae_pretraining_single_modality(backbone: str,
                                        decoder_settings: dict = None,
                                        patch_size: Union[int, list, tuple] = 16,
                                        crop_size=224,
                                        in_channels=1,
                                        drop_path_rate=0.0,
                                        use_cls_token=False,
                                        pretrained=False,
                                        **kwargs) -> MAE:
    printlog(f"warning: ignoring unused {kwargs}")

    ######################################## Prep Settings #############################################################
    backbone_settings = BACKBONES[backbone]
    embed_dim = backbone_settings["embed_dim"]
    depth = backbone_settings["depth"]
    num_heads = backbone_settings["num_heads"]
    patch_size = [patch_size, patch_size] if type(patch_size) == int else patch_size
    crop_size = [crop_size, crop_size] if type(crop_size) == int else crop_size
    out_features = np.prod(patch_size) * in_channels  # t * p_h * p_w


    if len(patch_size) == 2:
        patch_embed_type = "linear"
        patch_embed_params = []
    elif len(patch_size) == 3:  # case of 3D (BCTHW) input
        patch_embed_type = "linear"
        if len(crop_size) == 2:
            patch_embed_params = [PadIm2Video(ntimes=patch_size[0], pad_type="repeat")]
            printlog(f"Given a 3d patch_size [{patch_size}] but crop_size is 2D [{crop_size}] "
                     f"so using PadIm2Vdeo to make input BCTHW")
        else:
            patch_embed_params = []
    else:
        raise ValueError(f"patch_size must be of length 2 or 3, got {len(patch_size)} instead {patch_size}")

    # default decoder settings
    # default assumes alternating training with shared encoder but separate decoders hence "shared" set to False

    if backbone == "vit_tiny":
        decoder_settings_tiny = {"depth": 3, "decoder_embed_dim": 192, "qkv_bias": True, "decoder_num_heads": 3}
        printlog(f"backbone is {backbone}, so decoder_settings ovveriden from {decoder_settings} "
                 f"from {decoder_settings_tiny}")
        decoder_settings = decoder_settings_tiny

    default_decoder_settings = {"depth": 4, "decoder_embed_dim": 384, "qkv_bias": True, "decoder_num_heads": 16}
    if decoder_settings is None:
        printlog(f"decoder_settings not provided, using default {default_decoder_settings}")
        decoder_settings = default_decoder_settings
    # if partial decoder settings were provided then fill in the rest with defaults
    if "depth" not in decoder_settings:
        decoder_settings["depth"] = default_decoder_settings["depth"]
        printlog(f"decoder_settings['depth'] not provided, using default {default_decoder_settings['depth']}")
    if "decoder_embed_dim" not in decoder_settings:
        decoder_settings["decoder_embed_dim"] = default_decoder_settings["decoder_embed_dim"]
        printlog(f"decoder_settings['decoder_embed_dim'] not provided, using default "
                 f"{default_decoder_settings['decoder_embed_dim']}")
    if "qkv_bias" not in decoder_settings:
        decoder_settings["qkv_bias"] = default_decoder_settings["qkv_bias"]
        printlog(f"decoder_settings['qkv_bias'] not provided, using default {default_decoder_settings['qkv_bias']}")
    if "decoder_num_heads" not in decoder_settings:
        decoder_settings["decoder_num_heads"] = default_decoder_settings["decoder_num_heads"]
        printlog(f"decoder_settings['decoder_num_heads'] not provided, using default "
                 f"{default_decoder_settings['decoder_num_heads']}")

    ######################################## summary ##################################################################
    printlog(f"building {backbone} for pretraining with\n"
             f"out_features:{out_features} (inferred as in_channels * patch_size^2) \n"
             f"patch_size: {patch_size}\n"
             f"crop_size: {crop_size}\n"
             f"in_channels: {in_channels}\n"
             f"embed_dim: {embed_dim}\n"
             f"depth:{depth}\n"
             f"num_heads: {num_heads}\n"
             f"drop_path_rate: {drop_path_rate}\n"
             f"use_cls_token: {use_cls_token}\n"
             f"pretrained: {pretrained}\n"
             f"decoder_settings: {decoder_settings}\n")

    ####################################### Model init #################################################################
    attention_target = partial(
            Attention,
            attn_drop=0,
            num_heads=num_heads,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        )
    enc = VisionTransformer(
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=4,  # note: this is fixed for all ViT sizes
        attn_target=attention_target,
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        drop_path_type="progressive",
        classifier_feature="global_pool",
        use_cls_token=use_cls_token,
        patch_embed_type=patch_embed_type,
        patch_embed_params_list=patch_embed_params,
        layer_norm_eps=1e-6,
        masked_image_modeling=True,
        mask_token_embed_dim=None,
    )

    dec = Decoder(attn_target=partial(Attention,
                                      num_heads=decoder_settings['decoder_num_heads'],
                                      qkv_bias=decoder_settings["qkv_bias"]),
                  decoder_depth=decoder_settings["depth"],
                  patches_layout=enc.patch_embed.patches_layout,
                  first_patch_idx=enc.first_patch_idx,
                  input_embed_dim=enc.embed_dim,  # encoder.embed_dim,
                  embed_dim=decoder_settings["decoder_embed_dim"],
                  mask_token_embed_dim=decoder_settings["decoder_embed_dim"],
                  share_pos_embed=False)

    # L = enc.patch_embed.num_patches  # L = H*W/(P^2)
    # linear layer takes B*L,C sequences of tokens and maps them to B*L,out_features
    # where out_features = P*P^2*C = (16*16*1) = 256 # out_features=t*P*P*3 (t=2)
    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=decoder_settings["decoder_embed_dim"], out_features=out_features),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=0.02),
    )
    model = MAE(enc, dec, head)
    return model


def vit_mae_pretraining_multi_modality(backbone: str,  # one of the keys in BACKBONES
                                       patch_embeders: List[str],
                                       modalities: List[str],
                                       # decoders: dict = None, # todo remove
                                       decoder_settings: dict = None,
                                       patch_size: Union[int, list, tuple] = 16,
                                       crop_size=224,
                                       in_channels=1,  # todo assume all is grayscale for now
                                       drop_path_rate=0.0,
                                       use_cls_token=False,
                                       use_modality_token=False,
                                       modality_token_mode='add',
                                       use_all_seq=False,
                                       pretrained=False) -> Union[MultiModalMAE, MultiModalMAESeq]:
    """ Multi-modality pretraining with shared encoder and separate/shared decoders """

    ######################################## Prep Settings #############################################################
    # shared-ViT encoder settings
    backbone_settings = BACKBONES[backbone]
    embed_dim = backbone_settings["embed_dim"]
    depth = backbone_settings["depth"]
    num_heads = backbone_settings["num_heads"]
    out_features = patch_size * patch_size * in_channels

    patch_size = [patch_size, patch_size] if type(patch_size) == int else patch_size
    crop_size = [crop_size, crop_size] if type(crop_size) == int else crop_size

    # num_modalities = 2
    # if patch_embeders is None:
    #     patch_embeders = ['linear'] * num_modalities
    # if modalities is None:
    #     modalities = ['OCT', 'FA']

    # default decoder settings
    # default assumes alternating training with shared encoder but separate decoders hence "shared" set to False
    default_decoder_settings = {"depth": 4,
                                "decoder_embed_dim": 384,
                                "qkv_bias": True,
                                "decoder_num_heads": 16,
                                "shared": False,
                                "cross_modal_attention": False,
                                "batch_masking": "symmetric"
                                }

    # this indicates whether % of masked tokens per modality are constant across the batch (symmetric masking)
    # if "asymmetric" then the % of masked tokens per modality varies across batch elements

    if decoder_settings is None:
        printlog(f"decoder_settings not provided, using default {default_decoder_settings}")
        decoder_settings = default_decoder_settings

    # if partial decoder settings were provided then fill in the rest with defaults
    if "shared" not in decoder_settings:
        decoder_settings["shared"] = default_decoder_settings["shared"]
        printlog(f"decoder_settings['shared'] not provided, using default {default_decoder_settings['shared']}")
    if "depth" not in decoder_settings:
        decoder_settings["depth"] = default_decoder_settings["depth"]
        printlog(f"decoder_settings['depth'] not provided, using default {default_decoder_settings['depth']}")
    if "decoder_embed_dim" not in decoder_settings:
        decoder_settings["decoder_embed_dim"] = default_decoder_settings["decoder_embed_dim"]
        printlog(f"decoder_settings['decoder_embed_dim'] not provided, using default "
                 f"{default_decoder_settings['decoder_embed_dim']}")
    if "qkv_bias" not in decoder_settings:
        decoder_settings["qkv_bias"] = default_decoder_settings["qkv_bias"]
        printlog(f"decoder_settings['qkv_bias'] not provided, using default {default_decoder_settings['qkv_bias']}")
    if "decoder_num_heads" not in decoder_settings:
        decoder_settings["decoder_num_heads"] = default_decoder_settings["decoder_num_heads"]
        printlog(f"decoder_settings['decoder_num_heads'] not provided, using default "
                 f"{default_decoder_settings['decoder_num_heads']}")
    if "cross_modal_attention" not in decoder_settings:
        decoder_settings["cross_modal_attention"] = default_decoder_settings["cross_modal_attention"]
        printlog(f"decoder_settings['cross_modal_attention'] not provided, using default "
                 f"{default_decoder_settings['cross_modal_attention']}")
    if "batch_masking" not in decoder_settings:
        decoder_settings["batch_masking"] = default_decoder_settings["batch_masking"]
        printlog(f"decoder_settings['batch_masking'] not provided, using default "
                 f"{default_decoder_settings['batch_masking']}")

    assert not (decoder_settings["shared"] and decoder_settings["cross_modal_attention"]), \
        "decoder_settings['shared'] and decoder_settings['cross_modal_attention'] cannot both be True instead got " \
        f"{decoder_settings['shared']} and f{decoder_settings['cross_modal_attention']}"

    ######################################## summary ##################################################################
    printlog(f"building {backbone} for pretraining with\n"
             f"out_features:{out_features} (inferred as in_channels * patch_size^2) \n"
             f"patch_size: {patch_size}\n"
             f"crop_size: {crop_size}\n"
             f"in_channels: {in_channels}\n"
             f"embed_dim: {embed_dim}\n"
             f"depth:{depth}\n"
             f"num_heads: {num_heads}\n"
             f"drop_path_rate: {drop_path_rate}\n"
             f"use_cls_token: {use_cls_token}\n"
             f"use_modality_token: {use_modality_token}\n"
             f"modality_token_mode: {modality_token_mode}\n"
             f"pretrained: {pretrained}\n"
             f"patch_embeders: {patch_embeders}\n"
             f"modalities: {modalities}\n"
             f"use_all_seq: {use_all_seq}\n"
             f"decoders_settings: {decoder_settings}")

    ####################################### Model init #################################################################
    if use_all_seq:
        transformer_class = MultimodalVisionTransformerSeq
    else:
        transformer_class = MultimodalVisionTransformer
    enc = transformer_class(
        modalities=modalities,
        img_size=crop_size,  # default if not used cannot initialize from checkpoint
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=num_heads,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        patch_embed_types=patch_embeders,
        masked_image_modeling=True,
        use_cls_token=use_cls_token,
        use_modality_token=use_modality_token,
        modality_token_mode=modality_token_mode,
        drop_path_rate=drop_path_rate,
        drop_path_type="progressive",
        drop_rate=0.0,
        classifier_feature="global_pool",
        learnable_pos_embed=False,
        layer_norm_eps=1e-6,
        mask_token_embed_dim=None
    )

    if decoder_settings["shared"]:
        # one decoder for the whole sequence that includes tokens from all modalities
        # assume all modalities have the same length hence the same patches_layout (so we can use "OCT" as reference)
        if use_all_seq and use_modality_token and modality_token_mode == "concat":
            raise NotImplementedError("use_all_seq and use_modality_token and modality_token_mode=concat not supported")

        dec = Decoder(attn_target=partial(Attention,
                                          num_heads=decoder_settings['decoder_num_heads'],
                                          qkv_bias=decoder_settings["qkv_bias"]),
                      decoder_depth=decoder_settings["depth"],
                      patches_layout=enc.patch_embeders[modalities[0]].patches_layout,
                      first_patch_idx=enc.first_patch_idx,
                      input_embed_dim=enc.embed_dim,  # encoder.embed_dim,
                      embed_dim=decoder_settings["decoder_embed_dim"],
                      mask_token_embed_dim=decoder_settings["decoder_embed_dim"],
                      use_modality_token=use_modality_token,
                      modalities=modalities,
                      use_all_seq=use_all_seq,
                      share_pos_embed=False)

        heads = nn.ModuleDict()
        for modality in modalities:
            heads[modality] = make_conv_or_linear(layer=torch.nn.Linear(
                in_features=decoder_settings["decoder_embed_dim"], out_features=out_features),
                init_bias=partial(torch.nn.init.zeros_),
                init_weight=partial(trunc_normal_, mean=0.0, std=0.02))

        model = MultiModalMAESeq(enc, dec, heads, modalities)

    else:
        # modality-specific decoders
        decoders = nn.ModuleDict()
        for modality in modalities:
            # here we assume that all modalities have the same patches_layout
            printlog(f"building decoder for modality {modality} setting first_patch_index: 0 "
                     f"(enc.first_patch_idx : {enc.first_patch_idx}) ")
            if decoder_settings["cross_modal_attention"]:
                decoders[modality] = DecoderCrossAttention(attn_target=partial(Attention,
                                                                               num_heads=decoder_settings['decoder_num_heads'],
                                                                               qkv_bias=decoder_settings["qkv_bias"]),
                                                           cross_attn_target=partial(CrossAttention,
                                                                                     num_heads=decoder_settings['decoder_num_heads'],
                                                                                     qkv_bias=decoder_settings["qkv_bias"]),
                                                           decoder_depth=decoder_settings["depth"],
                                                           patches_layout=enc.patch_embeders[modalities[0]].patches_layout,
                                                           first_patch_idx=0,
                                                           input_embed_dim=enc.embed_dim,  # encoder.embed_dim,
                                                           embed_dim=decoder_settings["decoder_embed_dim"],
                                                           mask_token_embed_dim=decoder_settings["decoder_embed_dim"],
                                                           use_modality_token=use_modality_token,
                                                           modality_token_mode=modality_token_mode,
                                                           modalities=[modality],
                                                           share_pos_embed=False)

            else:
                decoders[modality] = Decoder(attn_target=partial(Attention,
                                                                 num_heads=decoder_settings['decoder_num_heads'],
                                                                 qkv_bias=decoder_settings["qkv_bias"]),
                                             decoder_depth=decoder_settings["depth"],
                                             patches_layout=enc.patch_embeders[modalities[0]].patches_layout,
                                             first_patch_idx=0,
                                             input_embed_dim=enc.embed_dim,  # encoder.embed_dim,
                                             embed_dim=decoder_settings["decoder_embed_dim"],
                                             mask_token_embed_dim=decoder_settings["decoder_embed_dim"],
                                             use_modality_token=use_modality_token,
                                             modality_token_mode=modality_token_mode,
                                             modalities=[modality],
                                             share_pos_embed=False,
                                             batch_masking=decoder_settings["batch_masking"])
        # modality-specific heads
        heads = nn.ModuleDict()
        for modality in modalities:
            heads[modality] = make_conv_or_linear(layer=torch.nn.Linear(in_features=decoder_settings["decoder_embed_dim"],
                                                                        out_features=out_features),
                                                  init_bias=partial(torch.nn.init.zeros_),
                                                  init_weight=partial(trunc_normal_, mean=0.0, std=0.02))
        if use_all_seq:
            model = MultiModalMAESeq(enc, decoders, heads, modalities)
        else:
            model = MultiModalMAE(enc, decoders, heads, modalities)
    return model


# finetune configurations with single or multi modality downstream tasks

def vit_finetune_single_modality(out_features: int,
                                 backbone: str,
                                 patch_size: Union[int, list, tuple] = 16,
                                 crop_size=224,
                                 in_channels=1,
                                 drop_path_rate=0.0,
                                 use_cls_token=False,
                                 classifier_feature="global_pool",
                                 pretrained=True,
                                 internal_checkpoint_path: Union[str, None] = None, **kwargs) -> MAE:
    """ build MAE finetuning on single modality """

    ######################################## Prep Settings #############################################################
    printlog(f" extra kwargs: {kwargs} will be ignored")
    backbone_settings = BACKBONES[backbone]
    embed_dim = backbone_settings["embed_dim"]
    depth = backbone_settings["depth"]
    num_heads = backbone_settings["num_heads"]
    patch_size = [patch_size, patch_size] if type(patch_size) == int else patch_size
    crop_size = [crop_size, crop_size] if type(crop_size) == int else crop_size

    if len(patch_size) == 2:
        patch_embed_type = "linear"
        patch_embed_params = []
    elif len(patch_size) == 3:  # case of 3D (BCTHW) input
        patch_embed_type = "linear"
        if len(crop_size) == 2:
            patch_embed_params = [PadIm2Video(ntimes=patch_size[0], pad_type="repeat")]
            printlog(f"Given a 3d patch_size [{patch_size}] but crop_size is 2D [{crop_size}] "
                     f"so using PadIm2Vdeo to make input BCTHW")
        else:
            patch_embed_params = []

    else:
        raise ValueError(f"patch_size must be of length 2 or 3, got {len(patch_size)} instead {patch_size}")

    ######################################## summary ##################################################################
    printlog(f"building {backbone} for finetuning with\n"
             f"classifier_feature: {classifier_feature}\n"
             f"patch_size: {patch_size}\n"
             f"crop_size: {crop_size}\n"
             f"out_features:{out_features}\n"
             f"in_channels: {in_channels}\n"
             f"embed_dim: {embed_dim}\n"
             f"depth:{depth}\n"
             f"num_heads: {num_heads}\n"
             f"drop_path_rate: {drop_path_rate}\n"
             f"use_cls_token: {use_cls_token}\n"
             f"pretrained: {pretrained}\n"
             f"internal_checkpoint_path: {internal_checkpoint_path}")
    attention_layer = partial(Attention, attn_drop=0, num_heads=num_heads, proj_drop=0, qk_scale=False, qkv_bias=True)
    # printlog(f"initializing from internal pretrained checkpoint: {internal_checkpoint_path}")
    ####################################### Model init #################################################################
    enc = VisionTransformer(
        img_size=crop_size,
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=4,
        attn_target=attention_layer,
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        drop_path_type="progressive",
        classifier_feature=classifier_feature,
        use_cls_token=use_cls_token,
        patch_embed_type="linear",
        patch_embed_params_list=patch_embed_params,
        layer_norm_eps=1e-6,
        masked_image_modeling=False,
        mask_token_embed_dim=None
    )
    if classifier_feature == "mean_max_pool":
        head_embed_dim = embed_dim * 2  # does mean and max pool and cocnats thus doubling the embed_dim
    else:
        head_embed_dim = embed_dim

    head = make_conv_or_linear(
        layer=torch.nn.Linear(in_features=head_embed_dim, out_features=out_features),
        init_bias=partial(torch.nn.init.zeros_),
        init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05),
    )

    model = MAE(enc, None, head)
    return model


def mvit_finetune_single_modality(out_features: int,
                                  backbone: str,  # one of the keys in BACKBONES
                                  patch_embeders: List[str],
                                  modalities: List[str],
                                  patch_size: Union[int, list, tuple] = 16,
                                  crop_size=224,
                                  in_channels=1,  # todo assume all is grayscale for now
                                  drop_path_rate=0.0,
                                  use_cls_token=False,
                                  classifier_feature="global_pool",
                                  use_modality_token=False,
                                  # backwards compatibility with old multimodal models that use pos_embed_{modality}
                                  pretrained=False,
                                  **kwargs) -> MultiModalMAE:

    """ build MultiModalMAE finetuning on single modality """
    printlog(f"Warning: extra kwargs: {kwargs} will be ignored")

    ######################################## Prep Settings #############################################################
    # shared-ViT encoder settings
    backbone_settings = BACKBONES[backbone]
    embed_dim = backbone_settings["embed_dim"]
    depth = backbone_settings["depth"]
    num_heads = backbone_settings["num_heads"]
    patch_size = [patch_size, patch_size] if type(patch_size) == int else patch_size
    crop_size = [crop_size, crop_size] if type(crop_size) == int else crop_size

    ######################################## summary ###################################################################
    printlog(f"building {backbone} for finetune_single_modality with\n"
             f"classifier_feature: {classifier_feature}\n"
             f"out_features:{out_features} (inferred as in_channels * patch_size^2) \n"
             f"patch_size: {patch_size}\n"
             f"crop_size: {crop_size}\n"
             f"in_channels: {in_channels}\n"
             f"embed_dim: {embed_dim}\n"
             f"depth:{depth}\n"
             f"num_heads: {num_heads}\n"
             f"drop_path_rate: {drop_path_rate}\n"
             f"use_cls_token: {use_cls_token}\n"
             f"use_modality_token: {use_modality_token}\n"
             f"pretrained: {pretrained}\n"
             f"patch_embeders: {patch_embeders}\n"
             f"finetune modalities: {modalities}")

    ####################################### Model init #################################################################
    enc = MultimodalVisionTransformer(
        modalities=modalities,
        img_size=crop_size,  # default if not used cannot initialize from checkpoint
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=num_heads,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        patch_embed_types=patch_embeders,
        masked_image_modeling=False,
        use_cls_token=use_cls_token,
        use_modality_token=use_modality_token,
        drop_path_rate=drop_path_rate,
        drop_path_type="progressive",
        drop_rate=drop_path_rate,
        classifier_feature=classifier_feature,
        learnable_pos_embed=False,
        mask_token_embed_dim=None
    )

    if classifier_feature == "mean_max_pool":
        head_embed_dim = embed_dim * 2  # does mean and max pool and cocnats thus doubling the embed_dim
    else:
        head_embed_dim = embed_dim

    # L = trunk.patch_embed.num_patches  # L = H*W/(P^2) transformer sequence length
    # linear layer takes B*L,embed_dim sequences of tokens and maps them to B*L,out_features
    heads = nn.ModuleDict()
    for modality in modalities:
        heads[modality] = make_conv_or_linear(
            layer=torch.nn.Linear(in_features=head_embed_dim, out_features=out_features),
            init_bias=partial(torch.nn.init.zeros_),
            init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05)
        )
    model = MultiModalMAE(enc, None, heads, modalities)
    return model


def mvit_finetune_multi_modality(out_features: int,
                                 backbone: str,  # one of the keys in BACKBONES
                                 patch_embeders: List[str],
                                 modalities: List[str],
                                 patch_size: Union[int, list, tuple] = 16,
                                 crop_size=224,
                                 classifier_feature='global_pool_per_modality',
                                 in_channels=1,  # todo assume all is grayscale for now
                                 drop_path_rate=0.0,
                                 use_cls_token=False,
                                 use_modality_token=False,
                                 use_all_seq=False,
                                 pretrained=False,
                                 **kwargs) -> MultiModalMAESeq:
    """
    Config for finetuning MultiModalMAESeq on multiple modalities treated as a sequence on a many-to-one task
    """

    printlog(f"Warning: given kwargs {kwargs} which are are unused")
    # shared-ViT encoder settings
    backbone_settings = BACKBONES[backbone]
    embed_dim = backbone_settings["embed_dim"]
    depth = backbone_settings["depth"]
    num_heads = backbone_settings["num_heads"]

    patch_size = [patch_size, patch_size] if type(patch_size) == int else patch_size
    crop_size = [crop_size, crop_size] if type(crop_size) == int else crop_size

    printlog(f"building {backbone} for finetuning with\n"
             f"out_features:{out_features} () \n"
             f"patch_size: {patch_size}\n"
             f"crop_size: {crop_size}\n"
             f"in_channels: {in_channels}\n"
             f"embed_dim: {embed_dim}\n"
             f"depth:{depth}\n"
             f"num_heads: {num_heads}\n"
             f"drop_path_rate: {drop_path_rate}\n"
             f"classifier_feature {classifier_feature}\n" 
             f"use_cls_token: {use_cls_token}\n"
             f"use_modality_token: {use_modality_token}\n"
             f"pretrained: {pretrained}\n"
             f"patch_embeders: {patch_embeders}\n"
             f"modalities: {modalities}\n"
             f"use_all_seq: {use_all_seq}\n")

    masked_image_modeling = False  # here we finetune so no mim
    if use_all_seq:
        transformer_class = MultimodalVisionTransformerSeq
    else:
        raise NotImplementedError("use_all_seq: 'False' not supported ")
    enc = transformer_class(
        modalities=modalities,
        img_size=crop_size,  # default if not used cannot initialize from checkpoint
        patch_size=patch_size,
        in_chans=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        mlp_ratio=4,
        attn_target=partial(
            Attention,
            attn_drop=0,
            num_heads=num_heads,
            proj_drop=0,
            qk_scale=False,
            qkv_bias=True,
        ),
        patch_embed_types=patch_embeders,
        masked_image_modeling=masked_image_modeling,
        use_cls_token=use_cls_token,
        use_modality_token=use_modality_token,
        drop_path_rate=drop_path_rate,
        drop_path_type="progressive",
        drop_rate=0.0,
        classifier_feature=classifier_feature,
        learnable_pos_embed=False,
        # layer_scale_type=None,
        # layer_scale_init_value=0.1,
        layer_norm_eps=1e-6,
        mask_token_embed_dim=None,
    )
    # determining prediction head input dimension based on classifier_feature
    num_modalities = len(modalities)
    if classifier_feature == "global_pool_per_modality":
        in_embed_dim = num_modalities * embed_dim
    elif classifier_feature == "global_pool_from_modality":
        in_embed_dim = embed_dim
    elif classifier_feature == "mean_max_pool_per_modality":
        in_embed_dim = num_modalities * embed_dim * 2
    elif classifier_feature == "mean_max_pool_from_modality":
        in_embed_dim = embed_dim * 2
    else:
        in_embed_dim = embed_dim

    head = make_conv_or_linear(layer=torch.nn.Linear(in_features=in_embed_dim, out_features=out_features),
                               init_bias=partial(torch.nn.init.zeros_),
                               init_weight=partial(trunc_normal_, mean=0.0, std=0.02))
    model = MultiModalMAESeq(enc, None, head, modalities)
    return model






