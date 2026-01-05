"""The reproduced code to train control net."""
import os
import sys


sys.path.append(os.path.join(os.path.abspath(os.getcwd()), "ControlNet"))
import numpy as np
from utils import start, validation
from config_control import Config

if __name__ == "__main__":
    root_path = Config().train.base_path
    model_my = start(
        root_path,
        Config().train.model_path,
        Config().train.model_config_path,
        Config().train.log_path,
        Config().train.condition,
    )
    # fidscore, clipscores, fsim = validation(
    #     model_my, root_path, Config().train.log_path, Config().train.condition
    # )
    # np.savetxt(
    #     os.path.join(Config().train.log_path, "result.txt"),
    #     np.array([fidscore, clipscores, fsim]),
    # )
