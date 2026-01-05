"""The reproduced code to train control net."""
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ControlNet"))

import numpy as np
from utils import start, validation


condition = "canny"

if __name__ == "__main__":
    root_path = "/scratch/dixiyao"
    model_my = start(
        root_path,
        "/controlnet/models/control_sd15_" + condition + ".ckpt",
        "./AttackBlackControlNet/models/cldm_v15.yaml",
        "logs/estimated_client",
        condition,
    )
    fidscore, clipscores, fsim = validation(
        model_my,
        root_path,
        "logs/estimated_client/",
        condition,
    )
    np.savetxt(
        "logs/estimated_client/results.txt", np.array([fidscore, clipscores, fsim])
    )
