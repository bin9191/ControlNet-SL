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
        "./BackwardFreeControlNet/models/cldm_v15.yaml",
        "logs/clip_as_condition_fp16sigma",
        condition="canny",
    )
    fidscore, clipscores, fsim = validation(
        model_my, root_path, "logs/clip_as_condition_fp16sigma/", condition="canny"
    )
    np.savetxt(
        "logs/clip_as_condition_fp16sigma/results_clip.txt",
        np.array([fidscore, clipscores, fsim]),
    )
