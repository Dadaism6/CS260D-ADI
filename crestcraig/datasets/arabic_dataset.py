import torch

import torchdatasets as td
from crestcraig.datasets.get_data import get_dataset
class ArabicIndexedDataset(td.Dataset):
    def __init__(self, args, split_type = 'train'):
        super().__init__()
        self.args = args
        self.dataset = get_dataset(args, split_type=split_type)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        input_ids = torch.tensor(data['input_ids'].squeeze())
        attention_mask = torch.tensor(data['attention_mask'].squeeze())
        label = torch.tensor(data['label'])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label,
            # Include other fields if needed
            'source': data['source'],
            'country': data['country'],
            'num_arabic_chars': data['num_arabic_chars'],
            'index': index
        }

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        self._cachers = []