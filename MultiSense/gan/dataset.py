"""
MultiSense GAN: PyTorch Dataset & DataLoader
Handles normalized pose sequences and one-hot class conditioning.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PoseSequenceDataset(Dataset):
    def __init__(self, X, y, num_classes=3):
        """
        Args:
            X: numpy array of shape (N, T, D) e.g. (N, 60, 132)
            y: numpy array of shape (N,) with integer class labels [0, 1, 2]
            num_classes: total number of distinct classes
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.num_classes = num_classes
        # One-hot encoded labels
        self.y_onehot = torch.nn.functional.one_hot(self.y, num_classes=num_classes).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.y_onehot[idx]


def get_dataloader(X, y, batch_size=32, shuffle=True):
    dataset = PoseSequenceDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=(len(dataset) > batch_size))
