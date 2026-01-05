"""Safe Control Net"""
import torch
from ControlNet.cldm.cldm import ControlLDM, ControlNet
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding


class OurControlLDM(ControlLDM):
    """Change condition to latent use CLIP."""

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
            hint = torch.cat(cond["c_concat"], 1)
            hint = 2 * hint - 1
            hint = self.first_stage_model.encode(hint)
            hint = self.get_first_stage_encoding(hint).detach()
            control = self.control_model(
                x=x_noisy, hint=hint, timesteps=t, context=cond_txt
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
            )

        return eps


class OurControlNet(ControlNet):
    """Our control"""

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        outs = []

        h = hint + x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
