"""Safe Control Net"""
import torch
from ControlNet.cldm.cldm import ControlLDM, ControlNet
from ControlNet.ldm.modules.diffusionmodules.util import timestep_embedding


def symsigmoid(x):
    "Symmetric sigmoid function $|x|*(2\sigma(x)-1)$"
    return torch.abs(x) * (2 * torch.nn.functional.sigmoid(x) - 1)


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

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=...,
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,
        transformer_depth=1,
        context_dim=None,
        n_embed=None,
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
    ):
        super().__init__(
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
            use_spatial_transformer,
            transformer_depth,
            context_dim,
            n_embed,
            legacy,
            disable_self_attentions,
            num_attention_blocks,
            disable_middle_self_attn,
            use_linear_in_transformer,
        )
        self.fix_noise = None

    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        outs = []

        h = hint + x.type(self.dtype)
        if self.training:
            h = symsigmoid(h)
            # here we add a fix noise
            if self.fix_noise is None:
                self.fix_noise = torch.randn(h.shape[1], h.shape[2], h.shape[3])
            h += self.fix_noise.repeat(h.shape[0], 1, 1, 1).to(h.device)
            # Here we need to quantizde fp16 and try it.
            h = h.half()
            h = h.to(torch.float32)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs
