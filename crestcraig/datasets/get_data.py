from torchvision import transforms
import os
import pandas as pd
import torch
from transformers import AutoTokenizer
def get_dataset(args, split_type = "train"):
    label2id = {"MSA": 0, "MGH": 1, "EGY": 2, "LEV": 3, "IRQ": 4, "GLF": 5}
    country2id ={'PL': 0, 'JO': 1, 'SY': 2, 'LB': 3, 'TN': 4, 'unknown': 5, 'IQ': 6, 'EG': 7, 'LY': 8, 'AE': 9, 'BH': 10, 'MA': 11,
     'OM': 12, 'KW': 13, 'SA': 14, 'DZ': 15, 'YE': 16, 'SD': 17, 'QA': 18, 'DJ': 19, 'SO': 20, 'MR': 21, 'MSA': 22}
    id2country = {v: k for k, v in country2id.items()}
    if args.dataset == 'Arabic':
        file_path = '/mnt/d/ucla/cs260D/CS260D-ADI/full_cleaned_data.tsv'

        # Read the dataset
        df = pd.read_csv(file_path, sep='\t')
        filtered_df = df[df['split'] == split_type]

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix")

        # Tokenization function
        # Tokenization and label conversion function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def tokenize_and_convert_labels(text, dialect):
            encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=15,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded_dict['input_ids'][0],
                'attention_mask': encoded_dict['attention_mask'][0],
                'label': label2id[dialect]
            }

        # Check for GPU availability


        # Apply tokenization and label conversion to the text data
        tokenized_data = [tokenize_and_convert_labels(text, dialect)
                          for text, dialect in zip(filtered_df['text'], filtered_df['dialect'])]

        # Create a new DataFrame
        tokenized_df = pd.DataFrame({
            'input_ids': [data['input_ids'] for data in tokenized_data],
            'attention_mask': [data['attention_mask'] for data in tokenized_data],
            'label': [data['label'] for data in tokenized_data],
            'source': filtered_df['source'],
            'country': [country2id[country] for country in filtered_df['country']],
            'num_arabic_chars': filtered_df['num_arabic_chars']
        })

        return tokenized_df

    else:
        raise NotImplementedError(f'Unknown dataset: {args.dataset}')