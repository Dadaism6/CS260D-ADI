from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from robust_training import get_group_accuracy, get_accuracy
from robust_training import create_datasets
import json


if __name__=='__main__':
    df = pd.read_csv('./full_cleaned_data.tsv',sep='\t')
    tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix")
    id2label = {0: "MSA",1: "MGH",2: "EGY",3: "LEV",4: "IRQ",5: "GLF"}
    label2id = {"MSA":0,"MGH":1,"EGY":2,"LEV":3,"IRQ":4,"GLF":5}
    unique_countries = df['country'].unique()
    country2id = {c:i for i,c in enumerate(list(unique_countries))}
    id2country = {i:c for i,c in enumerate(list(unique_countries))}
    _,_,testset = create_datasets(df, tokenizer,label2id,country2id)
    model = AutoModelForSequenceClassification.from_pretrained(
      'spare_last'#, num_labels=6, id2label=id2label, label2id=label2id
    )
    group_accuracy = get_group_accuracy(model, testset,128,"cuda")
    
    english_accuracy = {
        "{}_{}".format(id2label[d],id2country[c]):val for (d,c),val in group_accuracy.items()
    }

    worst_accuracy = min(list(group_accuracy.values()))
    avg_accuracy = sum(list(group_accuracy.values()))/len(list(group_accuracy.values()))
    best_accuracy = max(list(group_accuracy.values()))
    group_weights = testset.group_weights
    english_weights = {
        "{}_{}".format(id2label[d],id2country[c]):val for (d,c),val in group_weights.items() if val != 0
    }
    from torch.utils.data import DataLoader
    testloader = DataLoader(testset, 128)
    accuracy = get_accuracy(model, testloader, 'cuda')
    
    result = dict(
        group_accuracy = english_accuracy,
        group_weights = english_weights,
        worst_accuracy = worst_accuracy,
        avg_accuracy = avg_accuracy,
        best_accuracy = best_accuracy,
        accuracy = accuracy
    )

    with open("spare_last_evalution.json", 'w') as f:
        json.dump(result,f)
    
    
