import torch

import torchdatasets as td
from crestcraig.datasets.get_data import get_dataset
class ArabicIndexedDataset(td.Dataset):
    def __init__(self, args, split_type = 'train'):
        super().__init__()
        self.args = args
        self.dataset = get_dataset(args, split_type=split_type)
        self.label2id = {"MSA": 0, "MGH": 1, "EGY": 2, "LEV": 3, "IRQ": 4, "GLF": 5}
        self.country2id = {'PL': 0, 'JO': 1, 'SY': 2, 'LB': 3, 'TN': 4, 'unknown': 5, 'IQ': 6, 'EG': 7, 'LY': 8, 'AE': 9,
                      'BH': 10, 'MA': 11,
                      'OM': 12, 'KW': 13, 'SA': 14, 'DZ': 15, 'YE': 16, 'SD': 17, 'QA': 18, 'DJ': 19, 'SO': 20,
                      'MR': 21, 'MSA': 22}
        self.id2country = {v: k for k, v in country2id.items()}

    def __getitem__(self, index):
        data = self.dataset.iloc[index]

        # Check if input_ids and attention_mask are already tensors
        input_ids = data['input_ids']
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids.squeeze())

        attention_mask = data['attention_mask']
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask.squeeze())

        # Check if label is already a tensor
        label = data['label']
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

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
    def group_partition(self):
        partition_keys = {
            (self.label2id[label], self.country2id[country]):[] for label in self.label2id.keys() for country in self.country2id.keys()
        }
        for i in range(len(self.df)):
            label,country = self.df.iloc[i]['dialect'], self.df.iloc[i]["country"]
            partition_keys[(label, country)].append(i)
        return partition_keys