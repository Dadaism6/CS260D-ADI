from torchvision import transforms
import pandas as pd
from transformers import AutoTokenizer
def get_dataset(args, train=True, train_transform=True):
    if args.dataset == 'Arabic':
        file_path = '../../full_cleaned_data.tsv'
        split_type = 'train' if train else 'test'  # Adjust as necessary

        # Read the dataset
        df = pd.read_csv(file_path, sep='\t')
        filtered_df = df[df['split'] == split_type]

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix")

        # Define the transformation function for tokenization
        def transform(text):
            return tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                truncation=True,
                return_tensors='pt'
            ).data

        # Apply transformation to the text data
        filtered_df['transformed_text'] = filtered_df['text'].apply(transform)

        return filtered_df
    else:
        raise NotImplementedError(f'Unknown dataset: {args.dataset}')