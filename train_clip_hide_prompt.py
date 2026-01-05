"""The reproduced code to train control net."""
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ControlNet"))

import numpy as np
from utils import start, validation


if __name__ == "__main__":
    root_path = "/scratch/dixiyao"
    model_my = start(
        root_path,
        "/controlnet/models/control_sd15_ini.ckpt",
        "./BackwardFreeControlNet/models/cldm_v15_hide_prompt.yaml",
        "logs/clip_hide_prompt_prettyaux",
        condition="canny",
    )
    fidscore, clipscores, fsim = validation(
        model_my,
        root_path,
        "logs/clip_hide_prompt_prettyaux/",
        condition="canny",
    )
    np.savetxt(
        "logs/clip_hide_prompt_prettyaux/results_clip.txt",
        np.array([fidscore, clipscores, fsim]),
    )
