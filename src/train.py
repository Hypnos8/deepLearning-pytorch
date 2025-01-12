import os.path
from pathlib import Path

import torch
import TrainHelper
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 32
TRAIN_SIZE = 0.80
LEARNING_RATE = 1e-3
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 50
WEIGHT_DECAY = 1e-5


trainHelper = TrainHelper.TrainHelper()
val_test_dl, train_dl = trainHelper.load_data(TRAIN_SIZE, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

# create an instance of our ResNet model
resNet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss_function = torch.nn.BCELoss()

# set up the optimizer (see t.optim)
optimizer = torch.optim.AdamW(resNet.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=WEIGHT_DECAY)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model=resNet, train_dl=train_dl, val_test_dl=val_test_dl, optim=optimizer, crit=loss_function,
                  early_stopping_patience=EARLY_STOPPING_PATIENCE, cuda=True)

# go, go, go... call fit on trainer
result = trainer.fit(epochs=EPOCHS)
trainHelper.plot_results(result)

