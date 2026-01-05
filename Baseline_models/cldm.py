"""Safe Control Net"""
import math
import numpy as np
import torch
import einops
from plato.utils import unary_encoding
from ControlNet.cldm.cldm import ControlNet
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding
from config_control import Config


def PatchShuffle(x):
    "Patch Shuffle function"
    patch_num = x.shape[2] // Config().train.patch_size
    x = einops.rearrange(
        x,
        "b c (h1 p1) (w1 p2)->b (h1 w1) (p1 p2 c)",
        p1=Config().train.patch_size,
        p2=Config().train.patch_size,
    )

    for bs in range(x.shape[0]):
        # random permutation
        x[bs] = x[bs][torch.randperm(x.shape[1]), :]
    x = einops.rearrange(
        x,
        "b (h1 w1) (p1 p2 c)->b c (h1 p1) (w1 p2)",
        p1=Config().train.patch_size,
        p2=Config().train.patch_size,
        h1=patch_num,
    )
    return x


def dp_gaussian(x, configuration):
    "Local differential privacy with additive noise"
    scale = (
        2
        * math.log(1.25 / configuration.delta)
        * configuration.sensitivity**2
        / configuration.epsilon**2
    )
    return x + torch.tensor(np.random.normal(0, scale, x.shape)).to(x.dtype).to(
        x.device
    )


def dp_random_response(x, configuration):
    "Local differential privacy with random response"
    logits = unary_encoding.encode(x)
    epsilon = configuration.epsilon
    logits = (
        torch.tensor(unary_encoding.randomize(logits.detach().cpu().numpy(), epsilon))
        .to(x.dtype)
        .to(x.device)
    )
    return logits


class OurControlNet(ControlNet):
    """Our control"""

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)

        h = self.input_blocks[0](h, emb, context)
        h += guided_hint
        # Baselines methods are implemented here.
        if (
            hasattr(Config().train, "use_patch_shuffling")
            and Config().train.use_patch_shuffling
        ):
            h = PatchShuffle(h)
        elif hasattr(Config().train, "dp"):
            if hasattr(Config().train.dp, "gaussian"):
                h = dp_gaussian(h, Config().train.dp.gaussian)
            elif hasattr(Config().train.dp, "rr"):
                h = dp_random_response(h, Config().train.dp.rr)
        outs.append(self.zero_convs[0](h, emb, context))

        for module, zero_conv in zip(self.input_blocks[1:], self.zero_convs[1:]):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
