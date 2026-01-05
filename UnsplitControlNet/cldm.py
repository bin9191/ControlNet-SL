"""Safe Control Net"""
from ControlNet.cldm.cldm import ControlLDM
from ControlNet.ldm.util import instantiate_from_config
from ControlNet.ldm.models.diffusion.ddpm import disabled_train


class UnsplitControlLDM(ControlLDM):
    """Change condition to latent use CLIP."""

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_config = config
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
