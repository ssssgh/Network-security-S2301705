import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class NB15Dataset(Dataset):
    def __init__(self, file_path, scalar_path, scaling_method, benign_only=True):
        self.data = pd.read_csv(file_path)

        # split X, y
        self.X = self.data.iloc[:, :-2].to_numpy(dtype=np.float32)
        self.attacks = self.data["attack_cat"].to_numpy(dtype=object)
        self.y = self.data["Label"].to_numpy(dtype=np.float32)
        print(f"benign flows: {len(self.y) - self.y.sum()}\t attack flows: {self.y.sum()}")

        # normalize
        scalar = np.load(scalar_path)
        if scaling_method == "minmax":
            x_max = scalar[0]
            x_min = scalar[1]
            self.X = (self.X - x_min) / (x_max - x_min + 1e-8)
        elif scaling_method == "standard":
            x_mean = scalar[0]
            x_std = scalar[1]
            self.X = (self.X - x_mean) / (x_std + 1e-8)

        # filter out attacks if specified
        if benign_only:
            benign_idx = self.y == 0
            self.X = self.X[benign_idx]
            self.attacks = self.attacks[benign_idx]
            self.y = self.y[benign_idx]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.attacks[index]


if __name__ == "__main__":
    dataset = DataLoader(NB15Dataset("./data/train_set.csv", "./data/minmax_scalar.npy", only_benign=True),
                         shuffle=True, batch_size=64, drop_last=True)
    for X, y, label in dataset:
        print(X.size())
        print(y.size())
        print(label)
        break
