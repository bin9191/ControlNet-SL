"""Safe Control Net"""
import torch
from ControlNet.cldm.cldm import ControlLDM, ControlNet, ControlledUnetModel
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding


def symsigmoid(x):
    "Symmetric sigmoid function $|x|*(2\sigma(x)-1)$"
    return torch.abs(x) * (2 * torch.nn.functional.sigmoid(x) - 1)


class OurControlledUnetModel(ControlledUnetModel):
    """Only using prompts on the client"""

    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,
        only_mid_control=False,
        server_txt="",
        **kwargs
    ):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(
                timesteps, self.model_channels, repeat_only=False
            )
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            use_server_txt = False
            for module in self.input_blocks:
                if use_server_txt:
                    h = module(h, emb, server_txt)
                else:
                    h = module(h, emb, context)
                    if self.training:
                        use_server_txt = True
                hs.append(h)
            if self.training:
                h = self.middle_block(h, emb, server_txt)
            else:
                h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(
                h,
                emb,
                server_txt,
            )

        h = h.type(x.dtype)
        return self.out(h)


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
            control_server_txt = torch.zeros((x_noisy.shape[0], 1, 768)).to(self.device)
            if self.training:
                server_txt = torch.zeros((x_noisy.shape[0], 1, 768)).to(self.device)
            else:
                server_txt = cond_txt
            control = self.control_model(
                x=x_noisy,
                hint=hint,
                timesteps=t,
                context=control_server_txt,
            )
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(
                x=x_noisy,
                timesteps=t,
                context=cond_txt,
                control=control,
                only_mid_control=self.only_mid_control,
                server_txt=server_txt,
            )

        return eps


def symsigmoid(x):
    "Symmetric sigmoid function $|x|*(2\sigma(x)-1)$"
    return torch.abs(x) * (2 * torch.nn.functional.sigmoid(x) - 1)


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
