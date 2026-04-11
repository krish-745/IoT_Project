import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size=15, normalize=True, mu=None, sigma=None):
        self.window_size = window_size
        if normalize:
            if mu is None:
                self.mu    = data.mean(axis=0, keepdims=True)
                self.sigma = data.std(axis=0, keepdims=True) + 1e-8
            else:
                self.mu    = mu
                self.sigma = sigma
            data = (data - self.mu) / self.sigma
        else:
            self.mu    = np.zeros((1, data.shape[1]))
            self.sigma = np.ones((1, data.shape[1]))
        self.data = data.astype(np.float32)
        self.n_sensors = data.shape[1]
        self.n_samples = len(data) - window_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size]
        x = torch.tensor(x.T, dtype=torch.float32)
        y = torch.tensor(y,   dtype=torch.float32)
        return x, y


def build_datasets(train_df, test_df, window_size=15):
    train_arr = train_df.values.astype(np.float32)
    test_arr  = test_df.values.astype(np.float32)
    train_ds = TimeSeriesDataset(train_arr, window_size=window_size, normalize=True)
    test_ds  = TimeSeriesDataset(test_arr, window_size=window_size, normalize=True,
                                  mu=train_ds.mu, sigma=train_ds.sigma)
    return train_ds, test_ds
