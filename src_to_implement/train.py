import os.path
from pathlib import Path

import torch


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

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
data_path = Path(os.path.join('data.csv'))
data_df = pd.read_csv(data_path, sep=';')
# filename;crack;inactive

X_train, x_test, y_train, y_test = train_test_split(data_df['filename'], data_df[['crack', 'inactive']], train_size=TRAIN_SIZE, stratify=data_df[['crack', 'inactive']], random_state=50)

train = X_train.to_frame()
train['crack'] = y_train['crack']
train['inactive'] = y_train['inactive']

test = x_test.to_frame()
test['crack'] = y_test['crack']
test['inactive'] = y_test['inactive']

# Class distribution

# result without stratify
# Frequency of cracks crack
# 0    1557
# 1     443
#
# Frequency of inactive inactive
# 0    1878
# 1     122



# result with stratify
# Frequency of cracks in train  crack
# 0    1246
# 1     354

# Frequency of inactive in train  inactive
# 0    1502
# 1      98

# Frequency of cracks in test  crack
# 0    311
# 1     89

# Frequency of inactive in test  inactive
# 0    376
# 1     24


# Classes have unequal distribution -> bad for training, model tends to classify images as 0 in this case, maybe use oversampling for the label 1? Undersampling may not be an option here
# Some Oversampling options
# - Synthetic Minority Oversampling TEchnique (SMOTE) https://arxiv.org/abs/1106.1813
# - ADAptive SYNthetic (ADASYN) https://ieeexplore.ieee.org/document/4633969


# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
val_test_dl = torch.utils.data.DataLoader(ChallengeDataset(test, 'val'), batch_size=BATCH_SIZE_TRAIN)
train_dl = torch.utils.data.DataLoader(ChallengeDataset(train, 'train'), batch_size=BATCH_SIZE_TEST, shuffle=True)


# create an instance of our ResNet model
resNet = model.ResNet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
loss_function = torch.nn.BCELoss()

# set up the optimizer (see t.optim)
#optimizer = torch.optim.Adam(resNet.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=1e-5)
optimizer = torch.optim.AdamW(resNet.parameters(), lr=LEARNING_RATE, amsgrad=True, weight_decay=WEIGHT_DECAY)

# create an object of type Trainer and set its early stopping criterion
trainer = Trainer(model=resNet, train_dl=train_dl, val_test_dl=val_test_dl, optim=optimizer, crit=loss_function, early_stopping_patience=EARLY_STOPPING_PATIENCE, cuda=False)

# go, go, go... call fit on trainer
result = trainer.fit(epochs=EPOCHS)

# plot the results
plt.plot(np.arange(len(result[0])), result[0], label='train loss')
plt.plot(np.arange(len(result[1])), result[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

print("\nBATCH_SIZE_TRAIN = ", BATCH_SIZE_TRAIN)
print("BATCH_SIZE_TEST = ", BATCH_SIZE_TEST)
print("TRAIN_SIZE = ", TRAIN_SIZE)
print("LEARNING_RATE = ", LEARNING_RATE)
print("EPOCHS = ", EPOCHS)
print("EARLY_STOPPING_PATIENCE = ", EARLY_STOPPING_PATIENCE)
print("WEIGHT_DECAY = ", WEIGHT_DECAY)
print("optimizer: ", optimizer.__class__.__name__)
