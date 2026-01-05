"""Add some utils"""
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ControlNet"))
import pickle

import cv2
import torch
import torchvision
from PIL import Image
from einops import rearrange
import numpy as np

from ControlNet.ldm.modules.diffusionmodules.util import zero_module
from dataset.dataset_coco import CoCoDataset
from customImageLogger import CustomImageLogger
from ControlNet.cldm.model import create_model, load_state_dict
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics.image import fid
from torchmetrics.multimodal import clip_score
import einops
from pytorch_msssim import SSIM
from config_control import Config

from ControlNet.annotator.util import HWC3
from ControlNet.annotator.uniformer import UniformerDetector
from ControlNet.annotator.openpose import OpenposeDetector
from ControlNet.annotator.midas import MidasDetector
from ControlNet.annotator.hed import HEDdetector
from ControlNet.annotator.mlsd import MLSDdetector
from ControlNet.annotator.canny import CannyDetector


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, : h_x - 1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, : w_x - 1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return ((x * (x - 1)) ** 2).mean()


def fsim(x, y, model):
    """The feature similarity"""
    x_ = model(x, None, None)
    x_ = torch.reshape(x_, (1, -1))
    y = torch.reshape(y, (1, -1))
    return torch.mean(torch.nn.functional.cosine_similarity(x_, y)).item()


def log_condition(img, path):
    "log image"
    grid = torchvision.utils.make_grid(img, nrow=4)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = np.clip(grid * 127.5 + 127.5, 0, 255).astype(np.uint8)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(grid).save(path)
    return img


def log_condition_hint(img, path):
    "log image"
    grid = torchvision.utils.make_grid(img, nrow=4)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = np.clip(grid * 255, 0, 255).astype(np.uint8)
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    Image.fromarray(grid).save(path)
    return img


def reinitialize_with_zero_final(weights: torch.nn.Module, zero_final=True):
    """Reinitialize model weights with final layer is zero convolution."""

    def weights_reset(m):
        if (
            isinstance(m, torch.nn.Conv1d)
            or isinstance(m, torch.nn.Conv2d)
            or isinstance(m, torch.nn.Conv3d)
        ):
            m.reset_parameters()

    weights.apply(weights_reset)
    if zero_final:
        zero_module(weights[-1])
    return weights


def test_model_client_size(model_structure):
    "Save the model and measure the size."
    model = create_model(model_structure).cpu()
    payload = model.control_model.input_hint_block.state_dict()
    print("Original Encoder size:", sys.getsizeof(pickle.dumps(payload)))
    payload = model.first_stage_model.state_dict()
    print("Pretrained Encoder size:", sys.getsizeof(pickle.dumps(payload)))
    payload = model.model.diffusion_model.input_blocks[0]
    print("Client stable diffusion size:", sys.getsizeof(pickle.dumps(payload)))


def test_inference_memory_usage(root_path, model_structure):
    "Test the memory usage in inference"
    model = create_model(model_structure).cpu()
    valdataset = CoCoDataset(
        root_path + "/coco/",
        "val",
        dataset_size=Config().utils.validation_dataset_size,
        condition="canny",
    )
    valdataloader = DataLoader(valdataset, num_workers=0, batch_size=1, shuffle=False)
    model.eval()
    model = model.cuda()
    for batch in valdataloader:
        with torch.no_grad():
            model.log_images(batch)
        break
    print(torch.cuda.max_memory_allocated())


def load_model(root_path, initial_model_name, model_structure):
    "Start the training."
    # Configs
    resume_path = root_path + initial_model_name
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_structure).cpu()
    weights = load_state_dict(resume_path, location="cpu")
    model_weights = model.state_dict()
    for key, value in weights.items():
        if not key in model_weights.keys():
            print(key)
        else:
            model_weights[key] = value
    model.load_state_dict(model_weights)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    return model


def start(root_path, initial_model_name, model_structure, log_dir, condition):
    "Start the training."
    batch_size = Config().utils.batch_size
    logger_freq = Config().utils.logger_freq
    model = load_model(root_path, initial_model_name, model_structure)

    # Misc
    dataset = CoCoDataset(
        root_path + "/coco/",
        "train",
        condition=condition,
        dataset_size=Config().utils.training_dataset_size,
    )
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    valdataset = CoCoDataset(
        root_path + "/coco/",
        "val",
        condition=condition,
    )
    valdataloader = DataLoader(
        valdataset, num_workers=0, batch_size=batch_size, shuffle=False
    )
    logger = CustomImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(
        gpus=1,
        precision=32,
        callbacks=[logger],
        default_root_dir=log_dir,
        max_epochs=Config().utils.training_epochs,
    )

    # Train!
    trainer.fit(model, dataloader, val_dataloaders=valdataloader)
    return model


def validation(model, root_path, log_dir, condition):
    """Validation process."""
    sim_cond_scores = 0
    metric_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    fidscore = fid.FrechetInceptionDistance()
    clipscore = clip_score.CLIPScore(
        model_name_or_path=root_path
        + "/controlnet/models/openai/clip-vit-large-patch14"
    )
    valdataset = CoCoDataset(
        root_path + "/coco/",
        "val",
        condition=condition,
    )
    valdataloader = DataLoader(
        valdataset, num_workers=0, batch_size=Config().utils.batch_size, shuffle=False
    )
    model.eval()
    model = model.cuda()
    count = 0
    for batch in valdataloader:
        count += 1
        with torch.no_grad():
            generate = model.log_images(
                batch,
                ddim_steps=20,
            )
            origin = batch["jpg"]
            hint = batch["hint"]
            origin = einops.rearrange(origin, "b h w c -> b c h w").clone()
            generate = (
                torch.clip(generate["samples_cfg_scale_9.00"] * 127.5 + 127.5, 0, 255)
                .to(torch.uint8)
                .detach()
                .cpu()
            )
            origin = (
                torch.clip(origin * 127.5 + 127.5, 0, 255)
                .to(torch.uint8)
                .detach()
                .cpu()
            )
            fidscore.update(generate, real=False)
            fidscore.update(origin, real=True)
            clipscore.update(generate, batch["txt"])
            generate_conditions = []
            for index_img in range(generate.shape[0]):
                generate_condition_img = generate[index_img]
                generate_condition_img = einops.rearrange(
                    generate_condition_img, "c h w -> h w c"
                )
                generate_condition_img = generate_condition_img.numpy()
                generate_condition_img = valdataset.process(generate_condition_img)
                generate_conditions.append(generate_condition_img.tolist())
            generate_conditions = torch.tensor(
                np.array(generate_conditions).astype(np.float32) / 255.0
            )
            sim_cond = metric_ssim(
                einops.rearrange(generate_conditions, "b h w c-> b c h w"),
                einops.rearrange(hint, "b h w c->b c h w"),
            ).item()
            sim_cond_scores += sim_cond
            if count % 100 == 0:
                print(
                    "FID, CLIP, SIM:",
                    fidscore.compute().item(),
                    clipscore.compute().item(),
                    sim_cond_scores / count,
                )
    sim_cond_scores /= len(valdataloader)
    return (
        fidscore.compute().detach().item(),
        clipscore.compute().detach().item(),
        sim_cond_scores,
    )


def process_img_each(condition, img_each):
    if condition == "random":
        mean = np.mean(img_each, axis=(0, 1))
        std = np.std(img_each, axis=(0, 1))
        noise = np.random.normal(mean, std, img_each.shape)
        noise = np.clip(noise * 127.5 + 127.5, 0, 255)
        return noise
    if condition == "scribble":
        detected_map = np.zeros_like(img_each, dtype=np.uint8)
        detected_map[np.min(img_each, axis=2) < 127] = 255
        return detected_map
    if condition in ["seg", "pose", "normal", "hough", "hed", "canny", "depth"]:
        operators = {
            "seg": UniformerDetector,
            "depth": MidasDetector,
            "pose": OpenposeDetector,
            "normal": MidasDetector,
            "hough": MLSDdetector,
            "hed": HEDdetector,
            "canny": CannyDetector,
        }
        height, width, _ = img_each.shape
        if condition == "canny":
            detected_map = operators[condition]()(img_each, 100, 200)
            detected_map = HWC3(detected_map)
            return detected_map
        else:
            if condition == "depth" or condition == "pose":
                detected_map, _ = operators[condition]()(img_each)
            elif condition == "hough":
                detected_map = operators[condition]()(img_each, 0.1, 0.1)
            elif condition == "normal":
                _, detected_map = operators[condition]()(img_each, 0.4)
                detected_map = detected_map[:, :, ::-1]
            else:
                detected_map = operators[condition]()(img_each)
                detected_map = HWC3(detected_map)
                detected_map = cv2.resize(
                    detected_map, (height, width), interpolation=cv2.INTER_NEAREST
                )
            detected_map = HWC3(detected_map)
            return detected_map


def to_condition(img, condition):
    "log image"
    img = img.detach().cpu()
    imgs = []
    for index in range(img.shape[0]):
        img_each = img[index]
        img_each = img_each.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
        img_each = np.clip(img_each * 127.5 + 127.5, 0, 255).astype(np.uint8)
        detected_map = process_img_each(condition, img_each)
        imgs.append(detected_map.tolist())
    imgs = torch.tensor(imgs)
    imgs = rearrange(imgs, "b h w c->b c h w")
    imgs = imgs.float() / 255.0
    return img


def calculate_epsilon():
    # epsilon=8.85
    beta0 = 1e-4
    k = 2e-5
    items = np.arange(1000)
    items = np.sqrt(1.0 / (k * items + beta0) - 1)
    items = np.sum(items) / 1000
    return items


if __name__ == "__main__":
    print(calculate_epsilon())
