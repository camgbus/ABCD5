"""Manually define a torch.utils.data.Dataset from an events dataframe.
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda

class PandasDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]