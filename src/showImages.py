import os.path
from pathlib import Path

import torch
import torchvision as tv
from data import ChallengeDataset
from matplotlib import pyplot as plt
import pandas as pd

data_path = Path(os.path.join('data.csv'))
data_df = pd.read_csv(data_path, sep=';').head(1)

val_test_dl = torch.utils.data.DataLoader(ChallengeDataset(data_df, 'train'), batch_size=1)

img = next(iter(val_test_dl))[0][0]

plt.imshow(tv.transforms.ToPILImage()(img))
plt.show()

