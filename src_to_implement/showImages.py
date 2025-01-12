import os.path
from pathlib import Path

import skimage
from skimage.io import imread

import torch
import torchvision as tv
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules



# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data_path = Path(os.path.join('data.csv'))
data_df = pd.read_csv(data_path, sep=';').head(1)
#print(data_df)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
val_test_dl = torch.utils.data.DataLoader(ChallengeDataset(data_df, 'train'), batch_size=1)

#img = next(iter(val_test_dl))[0][0]

#plt.imshow(tv.transforms.ToPILImage()(img))
#plt.show()


df = pd.read_csv(data_path, sep=';')

print("Frequency of cracks", df['crack'].value_counts())

print("Frequency of inactive", df['inactive'].value_counts())
