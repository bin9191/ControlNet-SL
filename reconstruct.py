"""The reproduced code to train control net."""
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ControlNet"))

from tqdm import tqdm
import torch
import einops
from torch.utils.data import DataLoader
from pytorch_msssim import SSIM

from dataset.dataset_fill50k import Fill50KDataset
from dataset.dataset_celeba import CelebA_condition
from dataset.dataset_coco import CoCoDataset
from dataset.dataset_omniglot import OmniglotDataset
from dataset.dataset_imagenet import ImageNet_condition
from ControlNet.cldm.model import create_model, load_state_dict
from utils import log_condition_hint, log_condition
from ReconstructClientModel.inversion import InversionClip, Inversion
from config_control import Config

dataset_name = Config().valid.valid_dataset
condition_name = Config().train.condition
only_valid = Config().valid.only_valid
target = Config().train.target
if target == "hint":
    log_func = log_condition_hint
    model_save_path = os.path.join(Config().valid.model_path, condition_name + ".pth")
else:
    log_func = log_condition
    model_save_path = os.path.join(Config().valid.model_path, "org.pth")
if hasattr(Config().train, "rec_channel") and Config().train.rec_channel == 320:
    inversion_model = Inversion
else:
    inversion_model = InversionClip

save_path = os.path.join(
    Config().train.log_path, condition_name + "/" + dataset_name + "/"
)

model_structure = Config().train.model_config_path


def start():
    "Start the training."
    # Configs
    if hasattr(Config().train, "origin") and Config().train.origin:
        client_model_path = os.path.join(
            Config().train.base_path,
            "controlnet/models/control_sd15_" + condition_name + "_estimated.pth",
        )
    else:
        client_model_path = os.path.join(
            Config().train.base_path,
            "controlnet/models/control_sd15_" + condition_name + ".pth",
        )
    resume_path = client_model_path
    batch_size = Config().utils.batch_size
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False
    device = "cuda"
    epochs = 1

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_structure).cpu()
    model.load_state_dict(load_state_dict(resume_path, location="cpu"))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset = CoCoDataset(
        os.path.join(Config().train.base_path, "coco/"),
        "train",
        condition=condition_name,
        dataset_size=50000,
    )
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    # Misc
    if dataset_name == "celeba":
        valdataset = CelebA_condition(
            os.path.join(Config().train.base_path, "celeba/"), "valid", condition_name
        )
    elif dataset_name == "coco":
        valdataset = CoCoDataset(
            os.path.join(Config().train.base_path, "coco/"),
            "val",
            condition=condition_name,
        )
    elif dataset_name == "fill50k":
        valdataset = Fill50KDataset(os.path.join(Config().train.base_path, "fill50k/"))
    elif dataset_name == "omniglot":
        valdataset = OmniglotDataset(
            root=os.path.join(Config().train.base_path, "omniglot"),
            condition=condition_name,
        )
    elif dataset_name == "ImageNet":
        valdataset = ImageNet_condition(
            path=os.path.join(Config().train.base_path, "tiny-ImageNet-200"),
            split="test",
            task=condition_name,
        )
    else:
        print("Dataset does not exist")
        raise ValueError
    valdataloader = DataLoader(
        valdataset, num_workers=0, batch_size=batch_size, shuffle=False
    )
    controlnet_client = model
    inversion_net = inversion_model()
    if only_valid:
        inversion_net.load_state_dict(
            torch.load(
                model_save_path,
                map_location="cpu",
            )
        )

    controlnet_client = controlnet_client.to(device)
    inversion_net = inversion_net.to(device)

    # Train!
    for _ in range(epochs):
        if not only_valid:
            train(
                controlnet_client,
                inversion_net,
                dataloader,
                valdataloader,
                model_save_path,
                save_path,
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        valid(
            controlnet_client,
            inversion_net,
            valdataloader,
            save_path,
            prefix="_val",
        )


def train(
    client_model: torch.nn.Module,
    rec_model: torch.nn.Module,
    dataloader,
    valdataloder,
    path,
    path_img,
    device="cuda",
):
    """Train the model."""
    rec_model.train()
    client_model.train()
    opt = torch.optim.AdamW(rec_model.parameters(), lr=1e-5)
    tbar = tqdm(dataloader)
    for index, batch in enumerate(tbar):
        with torch.no_grad():
            condition = batch[target].to(device)
            condition = einops.rearrange(condition, "b h w c -> b c h w")
            feature = client_model.shared_step(batch)
        reconstruction = rec_model(feature)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(reconstruction, condition)
        loss.backward()
        opt.step()
        tbar.set_description(f"MSE: {loss.item()}")
        tbar.refresh()
        if index % 100 == 0:
            log_func(reconstruction.detach().cpu(), path_img + "_rec.png")
            log_func(condition.detach().cpu(), path_img + "_cond.png")
            valid(
                client_model,
                rec_model,
                valdataloder,
                path_img,
                prefix="_val",
                in_middle=True,
            )
            client_model.train()
            client_model.attacker_train = True
            # save model
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            torch.save(rec_model.state_dict(), path)


@torch.no_grad()
def valid(
    client_model: torch.nn.Module,
    rec_model: torch.nn.Module,
    dataloader,
    path,
    device="cuda",
    prefix="",
    in_middle=False,
):
    """Evaluate the model."""
    client_model.eval()
    client_model.attacker_train = False
    metric_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    rec_model.eval()
    mses = 0
    sims = 0
    tbar = tqdm(dataloader)
    for batch in tbar:
        condition = batch[target].to(device)
        condition = einops.rearrange(condition, "b h w c -> b c h w")
        feature = client_model.shared_step(batch)
        reconstruction = rec_model(feature)
        mse = torch.nn.functional.mse_loss(reconstruction, condition).item()
        ssim = metric_ssim(
            torch.clip(reconstruction, 0, 1), torch.clip(condition, 0, 1)
        ).item()
        mses += mse
        sims += ssim
        if in_middle:
            break
        tbar.set_description(f"MSE: {mse}, SSIM:{ssim}")
        tbar.refresh()
    batch_counts = len(dataloader)
    print("MSE: " + str(mses / batch_counts) + ", SSIM: " + str(sims / batch_counts))
    log_func(reconstruction.detach().cpu(), path + "_rec" + prefix + ".png")
    log_func(condition.detach().cpu(), path + "_cond" + prefix + ".png")
    with open(
        os.path.join(Config().train.log_path, "result_" + dataset_name + ".txt"), "w"
    ) as f:
        f.write(
            "MSE: " + str(mses / batch_counts) + ", SSIM: " + str(sims / batch_counts)
        )


if __name__ == "__main__":
    start()
