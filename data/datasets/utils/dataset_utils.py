import numpy as np
from torch.utils import data
import random

class StohasticSubsetDataset(data.Dataset):

    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        idx = random.randint(0, len(self.dataset))
        return self.dataset[idx]

class SubsetDataset(data.Dataset):

    def __init__(self, dataset, num_samples):
        self.dataset = dataset
        self.indices = np.random.randint(0, len(dataset), num_samples)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]