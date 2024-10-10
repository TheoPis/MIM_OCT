import torch
from torch import nn
from .distributed import is_distributed
__all__ = ["PolyakAverager"]


class PolyakAverager:

    def __init__(self, model: nn.Module, average_model: nn.Module, alpha: float = 0.99):
        self.alpha = alpha
        if is_distributed():
            self.model = model.module
        else:
            self.model = model
        self.polyak_enabled = average_model is not None
        self.average_model = average_model
        self.init_average_model()

    @torch.no_grad()
    def init_average_model(self):
        """Copy base model into average model."""

        if self.polyak_enabled:
            dst_dict = self.average_model.state_dict()
            for key, value in self.model.state_dict().items():
                dst_dict[key][...] = value
        else:
            self.average_model = self.model

    @torch.no_grad()
    def update(self):
        if self.polyak_enabled:
            """Update the average model with the contents of the base model."""
            dst_dict = self.average_model.state_dict()
            # print("ema update")
            src_dict = self.model.state_dict()
            for key, value in src_dict.items():
                dst = dst_dict[key]
                dst[...] = self.alpha * dst + (1 - self.alpha) * value

    @torch.no_grad()
    def __call__(self, x=None, **kwargs):
        if 'modality' in kwargs.keys():
            modality = kwargs['modality']
            return self.average_model(x, modality=modality)
        # fixme workaround to handle CLIP input that requires 'image_a' or 'image_b' as kwargs keys
        if 'image_a' in kwargs.keys():
            return self.average_model(image_a=kwargs['image_a'])
        elif 'image_b' in kwargs.keys():
            return self.average_model(image_b=kwargs['image_b'])
        else:
            return self.average_model(x)
