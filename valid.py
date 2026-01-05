"""The reproduced code to train control net."""
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ControlNet"))
import numpy as np
from utils import validation, load_model
from config_control import Config

if __name__ == "__main__":
    model_my = load_model(
        Config().train.base_path,
        Config().train.model_path,
        Config().train.model_config_path,
    )
    fidscore, clipscores, fsim = validation(
        model_my,
        Config().train.base_path,
        Config().train.log_path,
        Config().train.condition,
    )
    print(fidscore, clipscores, fsim)
    path = os.path.join(Config().train.log_path, "result.txt")
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    np.savetxt(
        path,
        np.array([fidscore, clipscores, fsim]),
    )
