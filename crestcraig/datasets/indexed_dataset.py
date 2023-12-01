import typing

import torchdatasets as td
from crestcraig.datasets.get_data import get_dataset


class IndexedDataset(td.Dataset):
    def __init__(self, args, train=True, train_transform=False):
        super().__init__()
        # self.dataset = get_dataset(args, train=train, train_transform=train_transform)
        self.dataset = None # load dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        self._cachers = []