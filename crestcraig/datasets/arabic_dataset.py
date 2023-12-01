import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from crestcraig.datasets.get_data import get_dataset
class ArabicIndexedDataset(Dataset):
    def __init__(self, args, split_type = 'train'):
        super().__init__()
        self.args = args
        self.dataset = get_dataset(args, split_type=split_type)

    def __getitem__(self, index):
        data = self.dataframe.iloc[index]
        input_ids = data['input_ids'].squeeze()
        attention_mask = data['attention_mask'].squeeze()
        label = data['label']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            # Include other fields if needed
            'source': data['source'],
            'dialect': data['dialect'],
            'country': data['country'],
            'num_arabic_chars': data['num_arabic_chars']
        }

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        self._cachers = []