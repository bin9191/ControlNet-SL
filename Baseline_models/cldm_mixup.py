"""Safe Control Net"""
import torch
import einops

from ControlNet.cldm.cldm import ControlLDM, ControlNet
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding


def mixup_process(tensor, key):
    key = key.to(tensor.device)
    return einops.einsum(key, tensor, "b1 b, b c h w->b1 c h w")


def generate_key_mixup(bs):
    enc_key = torch.exp(torch.randn((bs, bs)))
    for row_index in range(enc_key.shape[0]):
        enc_key[row_index] /= torch.sum(enc_key[row_index])
    dec_key = torch.inverse(enc_key)
    return enc_key, dec_key


class OurControlLDM(ControlLDM):
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond["c_crossattn"], 1)

        if cond["c_concat"] is None:
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=None,
                only_mid_control=self.only_mid_control,
            )
        else:
            enc_key, dec_key = generate_key_mixup(x_noisy.shape[0])
            control = self.control_model(
                x=x_noisy,
                hint=torch.cat(cond["c_concat"], 1),
                timesteps=t,
                context=cond_txt,
                mixkey=enc_key,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )
            eps = mixup_process(eps, dec_key)

        return eps


class OurControlNet(ControlNet):
    """Our control"""

    def forward(self, x, hint, timesteps, context, mixkey, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(self.dtype)

        h = self.input_blocks[0](h, emb, context)
        h += guided_hint
        # Baselines methods are implemented here.
        h = mixup_process(h, mixkey)

        outs.append(self.zero_convs[0](h, emb, context))

        for module, zero_conv in zip(self.input_blocks[1:], self.zero_convs[1:]):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
