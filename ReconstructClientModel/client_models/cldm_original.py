"""Safe Control Net"""
import math
import einops
import torch
import numpy as np
from plato.utils import unary_encoding
from ControlNet.cldm.cldm import ControlLDM, ControlNet
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding
from ControlNet.ldm.util import default
from config_control import Config


def mixup_process(tensor, key):
    key = key.to(tensor.dtype).to(tensor.device)
    return einops.einsum(key, tensor, "b1 b, b c h w->b1 c h w")


def generate_key_mixup(bs):
    enc_key = torch.exp(torch.randn((bs, bs)))
    for row_index in range(enc_key.shape[0]):
        enc_key[row_index] /= torch.sum(enc_key[row_index])
    dec_key = torch.inverse(enc_key)
    return enc_key, dec_key


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


class OrgControlLDM(ControlLDM):
    """Original ControlNet."""

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = x_start + self.q_sample(x_start=x_start, t=t, noise=noise)
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        hint = torch.cat(cond["c_concat"], 1)
        control = self.control_model(
            x=x_noisy,
            hint=hint,
            timesteps=t,
            context=cond_txt,
        )
        return control


class OrgControlNet(ControlNet):
    """Original ControlNet model."""

    # pylint:disable=unused-argument
    def forward(self, x, hint, timesteps, context, **kwargs):
        """Forward function."""
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

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
        elif hasattr(Config().train, "use_mixup") and Config().train.use_mixup:
            key, _ = generate_key_mixup(h.shape[0])
            h = mixup_process(h, key)

        return h
