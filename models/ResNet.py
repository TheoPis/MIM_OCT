import torch
from functools import partial
from torch import nn
from utils import printlog, DATASETS_INFO
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet50, resnet101
from timm.models.layers import trunc_normal_


def make_conv_or_linear(layer, init_weight=None, init_bias=None):
    if init_weight is not None:
        init_weight(tensor=layer.weight.data)
    if init_bias is not None:
        init_bias(tensor=layer.bias.data)
    return layer


class ResNetDetector(nn.Module):
    eligible_backbones = ['resnet18', 'resnet50', 'resnet101']
    torchvision2paper_resnet_layer_name_mapping = {"layer1": "C2", "layer2": "C3", "layer3": "C4", "layer4": "C5"}
    layers_strides_resnet = {"layer1": 0.5, "layer2": 0.25, "layer3": 0.125, "layer4": 0.125}

    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.backbone_name = config.get('backbone', 'resnet50')
        self.out_stride = config.get('out_stride', 16)
        self.experiment = experiment
        self.dataset = config['dataset']
        self.num_classes = DATASETS_INFO[self.dataset].NUM_CLASSES[experiment]

        # backbone settings
        self.replace_stride_with_dilation = self._set_striding_settings()
        self.backbone_cutoff = {'layer4': 'C5'}
        self.train_head_only = False  # set true if linear_probing and used by load_optiiser to only train head
        # we chop off fully connected layers from the backbone + load pretrained weights

        backbone_pretrained = config.get('pretrained', True)
        if self.backbone_name == 'resnet50':
            self.backbone = IntermediateLayerGetter(resnet50(pretrained=backbone_pretrained,
                                                             replace_stride_with_dilation=
                                                             self.replace_stride_with_dilation),
                                                    return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels

        elif self.backbone_name == 'resnet101':
            self.backbone = IntermediateLayerGetter(resnet101(pretrained=backbone_pretrained,
                                                              replace_stride_with_dilation=
                                                              self.replace_stride_with_dilation),
                                                    return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels

        printlog(f"ResNet backbone imagenet pretrained: {backbone_pretrained}")

        self.head = make_conv_or_linear(
            layer=torch.nn.Linear(in_features=self.backbone_out_channels, out_features=self.num_classes),
            init_bias=partial(torch.nn.init.zeros_),
            init_weight=partial(trunc_normal_, mean=0.0, std=2.0e-05))

        self.pooling = self.config.get('pooling', 'spatial_mean')
        if self.config['phase'] == 'linear_probing':
            self.get_linear_probing(use_bn=config.get('use_bn', False))

    def get_linear_probing(self, use_bn):
        """
        Sets the backbone to frozen. Head is initialized with trunc_normal
        """

        # freeze all but the head
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False
        for _, p in self.head.named_parameters():
            p.requires_grad = True
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_frozen_parameters = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        printlog(f"Going to use linear probing --> trainable parameters: {n_trainable_parameters} -- "
                 f"frozen parameters: {n_frozen_parameters}")

        # re-init head
        printlog(f"re-init head with trunc_normal")
        trunc_normal_(self.head.weight, std=0.01)
        self.train_head_only = True

        if use_bn:
            self.backbone.head = nn.Sequential(
                torch.nn.Sequential(torch.nn.BatchNorm1d(self.backbone.head.in_features,
                                                         affine=False,
                                                         eps=1e-6),
                                    self.backbone.head
                                    )
            )
            printlog("Using BN in linear probing head")

    def _set_striding_settings(self):
        assert (self.out_stride in [8, 16, 32])
        if self.out_stride == 8:
            layer_2_stride, layer_3_stride, layer_4_stride = False, True, True
        elif self.out_stride == 16:
            layer_2_stride, layer_3_stride, layer_4_stride = False, False, True
        else:
            layer_2_stride, layer_3_stride, layer_4_stride = False, False, False
        replace_stride_with_dilation = [layer_2_stride, layer_3_stride, layer_4_stride]
        assert (self.backbone_name in self.eligible_backbones), 'backbone must be in {}'.format(self.eligible_backbones)
        printlog(f"ResNetDetector backbone: {self.backbone_name}, out_stride: {self.out_stride},"
                 f" replace_stride_with_dilation at layers C3 C4 C5: {replace_stride_with_dilation}")
        return replace_stride_with_dilation

    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)
        backbone_features = self.backbone.forward(x)
        backbone_features = backbone_features['C5']
        if self.pooling == 'spatial_mean':
            backbone_features = backbone_features.mean(dim=(2, 3))
        else:
            raise NotImplementedError('Pooling type {} is not implemented'.format(self.pooling))
        logits = self.head(backbone_features)

        return logits

    def print_params(self):
        # just for debugging
        for w in self.state_dict():
            print(w, "\t", self.state_dict()[w].size())


if __name__ == '__main__':
    config = dict()
    config.update({'backbone': 'resnet50', 'pretrained': True, 'dataset': 'OctBiom'})
    config.update({'out_stride': 8})

    a = torch.ones(size=(1, 3, 416, 416))
    model = ResNetDetector(config, 1)
    # model.print_params()
    model.eval()
    b = model.forward(a)
    a = 1

