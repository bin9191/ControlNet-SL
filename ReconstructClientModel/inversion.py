"""Model inversion reconstruction net"""
import torch
from torch import nn
from ControlNet.ldm.modules.diffusionmodules.openaimodel import Upsample
from ControlNet.ldm.modules.diffusionmodules.util import conv_nd
from ControlNet.ldm.util import instantiate_from_config


class Inversion(torch.nn.Module):
    """Inversion network."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            conv_nd(2, 320, 320, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 320, 256, 3, padding=1),
            nn.SiLU(),
            Upsample(channels=256, use_conv=True, out_channels=96),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            Upsample(channels=96, use_conv=True, out_channels=32),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            Upsample(channels=32, use_conv=True, out_channels=16),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class InversionClip(torch.nn.Module):
    """Inversion network."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            Upsample(channels=4, use_conv=True, out_channels=96),
            nn.SiLU(),
            conv_nd(2, 96, 96, 3, padding=1),
            nn.SiLU(),
            Upsample(channels=96, use_conv=True, out_channels=32),
            nn.SiLU(),
            conv_nd(2, 32, 32, 3, padding=1),
            nn.SiLU(),
            Upsample(channels=32, use_conv=True, out_channels=16),
            nn.SiLU(),
            conv_nd(2, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, 16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class InversionClipDecoder(torch.nn.Module):
    """Inversion network."""

    def __init__(self, model_config, scale_factor) -> None:
        super().__init__()
        self.model = instantiate_from_config(model_config)
        self.scale_factor = scale_factor

    def forward(self, feature):
        feature = 1.0 / self.scale_factor * feature
        feature = self.model.decode(feature)
        return feature
