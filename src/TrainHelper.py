import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from src.data import ChallengeDataset


class TrainHelper:

    def load_data(self, train_size, batch_size_train, batch_size_test):
        """
             load the data from the csv file "data.csv" and perform a train-test-split
             The CSV file has the columns imagePath|Cracks|Interactive, where Cracks/inactive are 0/1, depending on whether
             there's a crack/intactive part
        """

        data_path = Path(os.path.join('data.csv'))
        data_df = pd.read_csv(data_path, sep=';')

        X_train, x_test, y_train, y_test = train_test_split(data_df['filename'], data_df[['crack', 'inactive']],
                                                            train_size=train_size,
                                                            stratify=data_df[['crack', 'inactive']],
                                                            random_state=50)
        train = X_train.to_frame()
        train['crack'] = y_train['crack']
        train['inactive'] = y_train['inactive']

        test = x_test.to_frame()
        test['crack'] = y_test['crack']
        test['inactive'] = y_test['inactive']

        # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
        val_test_dl = torch.utils.data.DataLoader(ChallengeDataset(test, 'val'), batch_size=batch_size_train)
        train_dl = torch.utils.data.DataLoader(ChallengeDataset(train, 'train'), batch_size=batch_size_test, shuffle=True)

        return val_test_dl, train_dl

    def plot_results(result):
        # plot the results
        plt.plot(np.arange(len(result[0])), result[0], label='train loss')
        plt.plot(np.arange(len(result[1])), result[1], label='val loss')
        plt.yscale('log')
        plt.legend()
        plt.savefig('losses.png')
