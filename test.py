import random
import os
import einops
import numpy as np
import torch
import cv2
from PIL import Image
from utils import load_model, process_img_each
from ControlNet.cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything

seed = 6666


def generate_image(model, control_dict, prompt_dict, types, condition):
    ddim_sampler = DDIMSampler(model)
    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    index = 0
    for control, prompt in zip(control_dict, prompt_dict):
        cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([prompt + ", " + a_prompt])],
        }
        un_cond = {
            "c_concat": [control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt])],
        }
        shape = (4, 512 // 8, 512 // 8)

        model.control_scales = [
            1.0
        ] * 13  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(
            20,
            1,
            shape,
            cond,
            verbose=False,
            eta=0.0,
            unconditional_guidance_scale=9.0,
            unconditional_conditioning=un_cond,
        )
        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        path = "examples/results/" + condition + "/" + types + "/" + str(index) + ".png"
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        cv2.imwrite(path, x_samples[0])
        index += 1


def depth_test(root_path, initial_model_name, model_structure, types):
    condition = "depth"
    model = load_model(root_path, initial_model_name, model_structure)
    model = model.cuda()
    model.eval()
    seed_everything(seed)

    control_dict = []
    img_dirct_need_process = [
        "examples/test_imgs/sd.png",
    ]
    for index, img_path in enumerate(img_dirct_need_process):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if index < 3:
            image = process_img_each(condition, image)
        path = (
            "examples/results/"
            + condition
            + "/condition/"
            + types
            + "/"
            + str(index)
            + ".png"
        )
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        cv2.imwrite(path, image)
        control = torch.from_numpy(image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        control_dict.append(control)
    prompt_dict = [
        "Stormtrooper's lecture",
    ]
    generate_image(model, control_dict, prompt_dict, types, condition)


def canny_test(root_path, initial_model_name, model_structure, types):
    condition = "canny"
    model = load_model(root_path, initial_model_name, model_structure)
    model = model.cuda()
    model.eval()
    seed_everything(seed)

    control_dict = []
    img_dirct_need_process = [
        "examples/test_imgs/bird.png",
        "examples/test_imgs/dog2.png",
        "examples/t2i/examples/canny/rabbit.png",
        "examples/t2i/examples/canny/toy_canny.png",
    ]
    for index, img_path in enumerate(img_dirct_need_process):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if index < 3:
            image = process_img_each(condition, image)
        control = torch.from_numpy(image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        control_dict.append(control)
    prompt_dict = [
        "a bird",
        "a cute dog",
        "a rabbit colored in blue with glasses",
        "a children toy",
    ]
    generate_image(model, control_dict, prompt_dict, types, condition)


def scribble_test(root_path, initial_model_name, model_structure, types):
    condition = "scribble"
    model = load_model(root_path, initial_model_name, model_structure)
    model = model.cuda()
    model.eval()
    seed_everything(seed)

    control_dict = []
    img_dirct_need_process = [
        "examples/t2i/examples/edit_cat/edge.png",
        "examples/t2i/examples/edit_cat/edge_2.png",
        "examples/test_imgs/bag_scribble.png",
        "examples/test_imgs/user_1.png",
        "examples/test_imgs/user_3.png",
    ]
    for _, img_path in enumerate(img_dirct_need_process):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = 255 - image
        control = torch.from_numpy(image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        control_dict.append(control)
    prompt_dict = ["a white cat", "a cartoon cat", "bag", "turtle", "hot air ballon"]
    generate_image(model, control_dict, prompt_dict, types, condition)


def seg_test(root_path, initial_model_name, model_structure, types):
    condition = "seg"
    model = load_model(root_path, initial_model_name, model_structure)
    model = model.cuda()
    model.eval()
    seed_everything(seed)

    control_dict = []
    img_dirct_need_process = [
        "examples/t2i/examples/seg/dinner.png",
        "examples/t2i/examples/seg/motor.png",
    ]
    for _, img_path in enumerate(img_dirct_need_process):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        control = torch.from_numpy(image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(1)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        control_dict.append(control)
    prompt_dict = ["dinner", "a motorcycle"]
    generate_image(model, control_dict, prompt_dict, types, condition)


if __name__ == "__main__":
    # scribble_test(
    #     "/data/dixi",
    #     "/controlnet/models/control_sd15_scribble.pth",
    #     "./ControlNet/models/cldm_v15.yaml",
    #     "org",
    # )
    seg_test(
        "models/",
        "control_sd15_seg.pth",
        "./ControlNet/models/cldm_v15.yaml",
        "org",
    )
    # depth_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_depth.pth",
    #     "./ControlNet/models/cldm_v15.yaml",
    #     "org",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/plato/results/controlnet/lightning_logs/version_18126359/checkpoints/epoch=0-step=1249-v9.ckpt",
    #     "./ControlNet/models/cldm_v15.yaml",
    #     "fl",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_autoencoder.pth",
    #     "./Baseline_models/cldm_v15_autoencoder.yaml",
    #     "autoencoder",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_safe.pth",
    #     "./OurSafeControlNet/models/cldm_v15.yaml",
    #     "safe",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_hide_prompt.pth",
    #     "./OurSafeControlNet/models/cldm_v15_hide_prompt.yaml",
    #     "hideprompt",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_hidepromptsafe_05_k.pth",
    #     "./OurSafeControlNet/models/cldm_v15_hide_prompt_safe_05_k.yaml",
    #     "hidepromptsafe_05_k",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_mixup.pth",
    #     "./Baseline_models/cldm_v15_mixup.yaml",
    #     "mixup",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_patchshuffling.pth",
    #     "./Baseline_models/cldm_v15.yaml",
    #     "patchshuffling",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_dp_rr.pth",
    #     "./Baseline_models/cldm_v15.yaml",
    #     "dp_rr",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_dp_gaussian_03.pth",
    #     "./Baseline_models/cldm_v15.yaml",
    #     "dp_gaussian_03",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_addnoise1.pth",
    #     "./Baseline_models/cldm_v15.yaml",
    #     "addnoise1",
    # )
    # canny_test(
    #     "/scratch/dixiyao",
    #     "/controlnet/models/control_sd15_canny_addnoise50.pth",
    #     "./Baseline_models/cldm_v15.yaml",
    #     "addnoise50",
    # )
