import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import get_dataset
class ArabicIndexedDataset(Dataset):
    def __init__(self, args, train=True, train_transform=False):
        super().__init__()
        self.args = args
        self.dataset = get_dataset(args, train=train, train_transform=train_transform)

    def __getitem__(self, index):
        if self.args.dataset == 'Arabic':
            row = self.dataset.iloc[index]
            data = row['transformed_text']  # This is the tokenized text
            target = row['label']  # Assuming there's a 'label' column in your dataset
        else:
            data, target = self.dataset[index]  # For other datasets

        return data, target, index

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        self._cachers = []