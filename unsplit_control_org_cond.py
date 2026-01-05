"""Use the unsplit attack to attack conditions from intermediate feature of ControlNet."""
import os
import sys
import copy

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ControlNet"))

import torch
from tqdm import trange
import einops
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM
from dataset.dataset_coco import CoCoDataset
from ControlNet.cldm.model import create_model, load_state_dict
from PromptsHiddenControlNet.inversion import InversionClipDecoder

from dataset.dataset_fill50k import Fill50KDataset
from dataset.dataset_celeba import CelebA_condition
from dataset.dataset_coco import CoCoDataset
from dataset.dataset_omniglot import OmniglotDataset
from dataset.dataset_imagenet import ImageNet_condition

from utils import (
    log_condition,
    log_condition_hint,
    fsim,
    TV,
    l2loss,
    reinitialize_with_zero_final,
)

dataset_name = "coco"


def main(model_name, condition_name):
    torch.manual_seed(0)
    "Start the training."
    # Configs
    resume_path = (
        "/scratch/dixiyao/controlnet/models/control_sd15_" + model_name + ".pth"
    )
    batch_size = 4
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    main_iters = 10000
    # lambda_tv = 0.01
    lambda_l2 = 1
    device = "cuda"
    # simulate place after
    add_z = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model("./UnsplitControlNet/cldm_v15.yaml").cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"), strict=False)
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    if dataset_name == "celeba":
        valdataset = CelebA_condition(
            "/scratch/dixiyao/celeba/", "valid", condition_name
        )
    elif dataset_name == "coco":
        valdataset = CoCoDataset(
            "/scratch/dixiyao/coco/",
            "val",
            condition=condition_name,
        )
    elif dataset_name == "fill50k":
        valdataset = Fill50KDataset("/scratch/dixiyao/fill50k/")
    elif dataset_name == "omniglot":
        valdataset = OmniglotDataset(
            root="/scratch/dixiyao/omniglot", condition=condition_name
        )
    elif dataset_name == "ImageNet":
        valdataset = ImageNet_condition(
            path="/scratch/dixiyao/tiny-ImageNet-200", split="test", task=condition_name
        )
    else:
        print("Dataset does not exist")
        raise ValueError
    valdataloader_decoder = DataLoader(
        valdataset, num_workers=0, batch_size=batch_size, shuffle=False
    )

    controlnet_client = copy.deepcopy(model.control_model.input_hint_block)
    attacked_model = reinitialize_with_zero_final(copy.deepcopy(controlnet_client))
    inversion_net_clip = InversionClipDecoder(
        model.first_stage_config, model.scale_factor
    )
    inversion_net_clip.eval()

    attacked_model_opt = torch.optim.Adam(
        attacked_model.parameters(), lr=0.001, amsgrad=True
    )
    attacked_model_noisy = reinitialize_with_zero_final(
        copy.deepcopy(model.control_model.input_blocks[0]), zero_final=False
    )
    attacked_model_noisy_opt = torch.optim.Adam(
        attacked_model_noisy.parameters(), lr=0.001, amsgrad=True
    )
    metric_ssim = SSIM(data_range=1.0, size_average=True, channel=3)

    controlnet_client.train()
    controlnet_client = controlnet_client.to(device)
    attacked_model.train()
    attacked_model = attacked_model.to(device)
    attacked_model_noisy.train()
    attacked_model_noisy = attacked_model_noisy.to(device)
    model = model.to(device)
    for batch in valdataloader_decoder:
        condition = batch["hint"]
        condition = einops.rearrange(condition, "b h w c -> b c h w")
        log_condition_hint(
            condition, "./logs/" + condition_name + "/unsplit_safe/condition.png"
        )
        condition = condition.to(device)
        feature = controlnet_client(condition, None, None)

        origin = batch["jpg"]
        origin = einops.rearrange(origin, "b h w c -> b c h w")
        log_condition(origin, "./logs/" + condition_name + "/unsplit_safe/origin.png")
        origin = origin.to(device)
        encoder_posterior = model.encode_first_stage(origin)
        z = model.get_first_stage_encoding(encoder_posterior).detach()
        noise = torch.randn_like(z)
        t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=device).long()
        x_noisy = model.q_sample(x_start=z, t=t, noise=noise)
        x_noisy_feature = model.control_model.input_blocks[0].to(device)(
            x_noisy, None, None
        )
        feature = feature + x_noisy_feature

        attacked_img = torch.nn.Parameter(
            (torch.clip(torch.zeros(condition.shape), -1, 1) * 0.5 + 0.5).to(device),
            requires_grad=True,
        )
        attacked_noise = torch.nn.Parameter(
            torch.randn_like(z).to(device),
            requires_grad=True,
        )
        attacked_z = torch.nn.Parameter(
            torch.randn_like(z).to(device),
            requires_grad=True,
        )
        input_opt = torch.optim.Adam(
            [attacked_img, attacked_noise, attacked_z], lr=0.001, amsgrad=True
        )
        tbar = trange(main_iters, desc="Bar desc", leave=True)
        for main_iter in tbar:
            input_opt.zero_grad()
            pred = attacked_model(attacked_img, None, None)
            pred += attacked_model_noisy(
                model.q_sample(x_start=attacked_z, t=t, noise=attacked_noise),
                None,
                None,
            )
            loss = (
                torch.nn.functional.mse_loss(pred, feature)
                # + lambda_tv * TV(attacked_img)
                + lambda_l2 * l2loss(attacked_img)
            )
            loss.backward(retain_graph=True)
            input_opt.step()
            loss1 = loss.item()

            attacked_model_noisy_opt.zero_grad()
            attacked_model_opt.zero_grad()
            pred = attacked_model(attacked_img, None, None)
            pred += attacked_model_noisy(
                model.q_sample(x_start=attacked_z, t=t, noise=attacked_noise),
                None,
                None,
            )
            loss = torch.nn.functional.mse_loss(pred, feature)
            loss.backward(retain_graph=True)
            attacked_model_opt.step()
            attacked_model_noisy_opt.step()

            ssim = metric_ssim(attacked_img, condition).item()
            mse_result = torch.nn.functional.mse_loss(attacked_img, condition).item()

            tbar.set_description(
                "Input stage loss %.4f, Model stage loss %.4f, SSIM %.4f, MSE %.4f"
                % (loss1, loss.item(), ssim, mse_result)
            )
            tbar.refresh()
            if main_iter % 1000 == 0 or main_iter == main_iters - 1:
                rec_org_img = inversion_net_clip(attacked_z.cpu())
                ssim_org = metric_ssim(
                    rec_org_img * 0.5 + 0.5, origin.cpu() * 0.5 + 0.5
                ).item()
                mse_result_org = torch.nn.functional.mse_loss(
                    rec_org_img, origin.cpu()
                ).item()
                log_condition(
                    rec_org_img.detach().cpu(),
                    "./logs//unsplit_safe/attack_canny_org%d.png" % main_iter,
                )
                log_condition_hint(
                    attacked_img.detach().cpu(),
                    "./logs//unsplit_safe/attack_canny%d.png" % main_iter,
                )
                print(
                    ssim,
                    mse_result,
                    ssim_org,
                    mse_result_org,
                )
        return (
            ssim,
            mse_result,
            ssim_org,
            mse_result_org,
        )


if __name__ == "__main__":
    results = []
    model_names = ["canny"]  # , "depth", "hed", "mlsd", "normal", "openpose"]
    condition_names = ["canny"]  # , "depth", "hed", "hough", "normal", "openpose"]
    for model_name, condition_name in zip(model_names, condition_names):
        results.append(main(model_name, condition_name))
    print(results)
