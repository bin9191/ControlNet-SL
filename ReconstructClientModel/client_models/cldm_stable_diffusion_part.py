"""Safe Control Net"""
import torch
from ControlNet.ldm.util import default
from ControlNet.cldm.cldm import ControlLDM
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding


def symsigmoid(x):
    "Symmetric sigmoid function $|x|*(2\sigma(x)-1)$"
    return torch.abs(x) * (2 * torch.nn.functional.sigmoid(x) - 1)


class OrgAttackControlLDM(ControlLDM):
    """Change condition to latent use CLIP."""

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = x_start + self.q_sample(x_start=x_start, t=t, noise=noise)
        context = torch.cat(cond["c_crossattn"], 1)
        with torch.no_grad():
            t_emb = timestep_embedding(
                t, self.model.diffusion_model.model_channels, repeat_only=False
            )
            emb = self.model.diffusion_model.time_embed(t_emb)
            h = x_noisy.type(self.model.diffusion_model.dtype)
            h = self.model.diffusion_model.input_blocks[0](h, emb, context)
        return h


class OurOrgAttackControlLDM(ControlLDM):
    """Change condition to latent use CLIP."""

    def __init__(
        self, control_stage_config, control_key, only_mid_control, *args, **kwargs
    ):
        super().__init__(
            control_stage_config, control_key, only_mid_control, *args, **kwargs
        )
        self.fix_noise = None
        self.attacker_train = True

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = x_start + self.q_sample(x_start=x_start, t=t, noise=noise)
        context = torch.cat(cond["c_crossattn"], 1)
        with torch.no_grad():
            t_emb = timestep_embedding(
                t, self.model.diffusion_model.model_channels, repeat_only=False
            )
            emb = self.model.diffusion_model.time_embed(t_emb)
            h = x_noisy.type(self.model.diffusion_model.dtype)
            h = self.model.diffusion_model.input_blocks[0](h, emb, context)
            if not self.attacker_train:
                h = symsigmoid(h)
                # here we add a fix noise
                if self.fix_noise is None:
                    self.fix_noise = torch.randn(h.shape[1], h.shape[2], h.shape[3])
                h += self.fix_noise.repeat(h.shape[0], 1, 1, 1).to(h.device)
            # Here we need to quantizde fp16 and try it.
            h = h.half()
            h = h.to(torch.float32)
        return h
