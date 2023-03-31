import torch
import numpy as np
from torch.utils.data import Dataset


class Processor(object):
    def __init__(self):
        super().__init__()

    def process(self, text):
        return list(text)


class HaoDaiFuDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
