import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.distributed import rank_zero_only
from ControlNet.cldm.logger import ImageLogger
from config_control import Config
# Custom Logger Class extended from ControlNet/cldm/logger.py
class CustomImageLogger(ImageLogger):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

            current_step = trainer.global_step
            checkpoint_idx = 5000 # checkpoint_idx = Config().utils.checkpoint_idx
            if (current_step % checkpoint_idx == 0):
                checkpoint_dir = f"./{Config().train.log_path}/checkpoints/"
                filename = f"epoch={pl_module.current_epoch}-step={current_step}.ckpt"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, filename)
                trainer.save_checkpoint(checkpoint_path)