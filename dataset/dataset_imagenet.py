"""Load CoCo dataset."""
import os
import csv
from collections import namedtuple
from typing import Optional
import copy

import cv2
from torchvision.datasets import ImageFolder
import torch
import numpy as np
from .dataset_basic import BasicDataset


CSV = namedtuple("CSV", ["header", "index", "data"])


class ImageNet_condition(BasicDataset):
    """Tiny ImageNet dataset"""

    def __init__(
        self,
        path,
        split,
        task,
    ):
        super().__init__(task=task)
        self.root = os.path.join(path, split)
        self.dataset = ImageFolder(self.root)

    def __getitem__(self, idx):
        path, _ = self.dataset.samples[idx]
        image = cv2.imread(path)
        mask = copy.deepcopy(image)
        image = cv2.resize(image, (512, 512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) - 127.5) / 127.5

        mask = cv2.resize(mask, (512, 512))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self.process(mask)
        mask = mask.astype(np.float32) / 255.0

        sentence = "Good image"
        return {"jpg": image, "hint": mask, "txt": sentence}

    def __len__(self) -> int:
        return len(self.dataset)
