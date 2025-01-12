import os.path
import random

import skimage.color
import torch.nn
from torch.utils.data import Dataset
from pathlib import Path
from skimage.io import imread
import numpy as np
import torchvision as tv
from torchvision.transforms import functional

#from torchvision.transforms.v2 import functional

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode, rotation=True, mirroring=True):
        self.rotation = rotation
        self.mirroring = mirroring
        self.mode = mode
        if not (mode == "train" or mode == "val"):
            raise ValueError
        if mode == "train":
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(),
                                                     # tv.transforms.RandomRotation(degrees=(0, 359)),
                                                     CustomAngleRotation(angles=[0, 90, 180, 270]),
                                                     tv.transforms.RandomHorizontalFlip(),
                                                     tv.transforms.RandomVerticalFlip(),
                                                     tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=train_mean, std=train_std)
                                                     ]
                                                    )
        elif mode == "val":
            self._transform = tv.transforms.Compose([tv.transforms.ToPILImage(), tv.transforms.ToTensor(),
                                                     tv.transforms.Normalize(mean=train_mean, std=train_std)])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        current_data = self.data.iloc[index]
        image_path = Path(current_data['filename'])
        image = skimage.color.gray2rgb(imread(image_path))
        image = self._transform(image)
        return image, np.array([current_data['crack'], current_data['inactive']])


class CustomAngleRotation:
    def __init__(self, angles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return functional.rotate(x, angle)
