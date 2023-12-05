from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoModel, AutoTokenizer, AutoModelForMaskedLM
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import Adam, SGD
import copy
import torch
from spuco.datasets.base_spuco_compatible_dataset import BaseSpuCoCompatibleDataset
import random
from typing import Iterator, List
import numpy as np
from torch.utils.data import Sampler
import json
from spuco.group_inference.spare_inference import SpareInference
from spuco.group_inference.cluster import ClusterAlg
from spuco.utils import convert_labels_to_partition
class ArabicDataset(BaseSpuCoCompatibleDataset):
    def __init__(self, dataframe, tokenizer, label2id, country2id):
        self.df = dataframe
        self.encodings = tokenizer(dataframe['text'].values.tolist(),truncation=True, padding=True)
        self.labels = dataframe['dialect'].apply(lambda x: label2id[x]).values.tolist()
        self.label2id = label2id
        self.country2id = country2id

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item, idx

    def __len__(self):
        return len(self.labels)
    
    @property
    def spurious(self):
        return [self.country2id[c] for c in list(self.df["country"])]
    @property
    def group_partition(self):
        partition_keys = {
            (self.label2id[label], self.country2id[country]):[] for label in self.label2id.keys() for country in self.country2id.keys()
        }
        for i in range(len(self.df)):
            label,country = self.label2id[self.df.iloc[i]['dialect']], self.country2id[self.df.iloc[i]["country"]]
            partition_keys[(label, country)].append(i)
        return partition_keys
    @property
    def group_weights(self):
        """
        Dictionary containing the fractional weights of each group
        """
        partition = self.group_partition
        total = len(self.labels)
        return {
            key: len(val)/total for key, val in partition.items()
        }
    
    def labels(self):
        return self.labels

    @property
    def num_classes(self):
        return len(self.label2id.keys())
    

def train(model,epochs,train_loader,dev_loader,optimizer,lr_scheduler, device):
    model.to(device)
    loss_log = {}
    accuracy_log = []
    max_accuracy = 0
    best_parameters = None
    for epoch in range(epochs):
        epoch_loss = []
        model.train()
        print("Start Training:")
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch,index in tepoch:
                optimizer.zero_grad()
                tepoch.set_description(f"Epoch {epoch}")
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids,attention_mask = attention_mask, labels = labels)
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                if lr_scheduler:
                  lr_scheduler.step()
                tepoch.set_postfix(loss=loss.item())
                epoch_loss.append(loss.item())
            loss_log[epoch] = epoch_loss
        
        model.eval()
        print("Evaluation:")
        num_right = 0
        num_items = 0
        with tqdm(dev_loader, unit="batch") as depoch:
            for batch,index in depoch:
                depoch.set_description(f"Epoch {epoch}")
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.no_grad():
                    output = model(input_ids,attention_mask)
                    logits = output.logits
                    predictions = torch.argmax(logits, dim = -1)
                    correct_num = (predictions == labels).sum()
                    num_right += correct_num
                    num_items += len(batch['labels'])
        accuracy = num_right / num_items
        print("accuracy= %.3f" %(accuracy))
        if accuracy.item() > max_accuracy:
            best_parameters = model.state_dict()
            max_accuracy = accuracy.item()
        accuracy_log.append(accuracy.item())
    return loss_log, accuracy_log, best_parameters
    
def get_accuracy(model, loader, device):
  num_items = 0
  num_correct = 0
  model.eval()
  step = 0
  with torch.no_grad():
    with tqdm(loader, unit="batch") as tepoch:
        for batch,index in tepoch:
          tepoch.set_description(f"Evaluating {step}")
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids,attention_mask = attention_mask)
          logits = outputs.logits
          predictions = torch.argmax(logits,dim = -1)
          right = (predictions == labels).sum()
          num = len(input_ids)
          num_items += num
          num_correct += right
          step += 1
    return (num_correct/num_items).item()
  
def get_accuracy_no_tqdm(model, loader, device):
  num_items = 0
  num_correct = 0
  model.eval()
  step = 0
  with torch.no_grad():
    for batch,index in loader:
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['labels'].to(device)
      outputs = model(input_ids,attention_mask = attention_mask)
      logits = outputs.logits
      predictions = torch.argmax(logits,dim = -1)
      right = (predictions == labels).sum()
      num = len(input_ids)
      num_items += num
      num_correct += right
      step += 1
  return (num_correct/num_items).item()

def get_group_accuracy(model, dataset,  batch_size, device):
    model.eval()
    accuracies = {}
    testloaders = {}
    group_partition = dataset.group_partition
    su = 0
    model.to(device)
    for key in group_partition.keys():
            if len(group_partition[key]) == 0:
                 continue
            su += len(group_partition[key])
            sampler = SubsetRandomSampler(group_partition[key])
            testloaders[key] = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    for key in tqdm(sorted(group_partition.keys()), "Evaluating group-wise accuracy", ):
        if len(group_partition[key]) == 0:
            continue
        accuracies[key] = get_accuracy_no_tqdm(model, testloaders[key],device)
        #print(f"Group {key} Accuracy: {accuracies[key]}")
    return accuracies

def generate_upsample_indices(model, dataloader, device):
  model.eval()
  step = 0
  indices = []
  with torch.no_grad():
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch,index in tepoch:
          tepoch.set_description(f"Evaluating {step}")
          input_ids = batch['input_ids'].to(device)
          attention_mask = batch['attention_mask'].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids,attention_mask = attention_mask)
          logits = outputs.logits
          predictions = torch.argmax(logits,dim = -1)
          masks = (predictions != labels).cpu()
          wrong_indices = index[masks]
          indices+=wrong_indices.tolist()
          step +=1
  return indices

class CustomIndicesSampler(Sampler[int]):
    """
    Samples from the specified indices (pass indices - upsampled, downsampled, group balanced etc. to this class)
    Default is no shuffle.
    """
    def __init__(
        self,
        indices: List[int],
        shuffle: bool = False,
    ):
        """
        Samples elements from the specified indices.

        :param indices: The list of indices to sample from.
        :type indices: list[int]
        :param shuffle: Whether to shuffle the indices. Default is False.
        :type shuffle: bool, optional
        """
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over the sampled indices.

        :return: An iterator over the sampled indices.
        :rtype: iterator[int]
        """
        if self.shuffle:
            random.shuffle(self.indices)
        return iter(self.indices)

    def __len__(self) -> int:
        """
        Returns the number of sampled indices.

        :return: The number of sampled indices.
        :rtype: int
        """
        return len(self.indices)
    
def create_upsample_dataloader(old_dataset, batch_size, error_indices, E):
   indices = list(range(len(old_dataset))) + E * error_indices
   copy_old = copy.deepcopy(old_dataset)
   loader = DataLoader(copy_old,batch_size, sampler = CustomIndicesSampler(indices,True))
   return loader

def create_datasets(df, tokenizer, label2id, country2id):
   '''
   End-to-End dataset creations. Create train, dev, test
   '''
   grouped_df = df.groupby('split')
   dfs = {name: group for name, group in grouped_df}
   train_df = dfs['train']#.sample(n=640)
   dev_df = dfs['dev']#.sample(n=32)
   test_df = dfs['test']#.sample(n=32)
   trainset = ArabicDataset(train_df, tokenizer,label2id, country2id)
   devset = ArabicDataset(dev_df, tokenizer, label2id, country2id)
   testset = ArabicDataset(test_df, tokenizer, label2id, country2id)
   return trainset, devset, testset

def baseline(config, datasets):
    model = AutoModelForSequenceClassification.from_pretrained(
      'CAMeL-Lab/bert-base-arabic-camelbert-mix', num_labels=6, id2label=id2label, label2id=label2id
    )
    if config['optimizer']=="Adam":
       optim = Adam(model.parameters(),config['lr'])
    else:
       optim = SGD(model.parameters(), config['lr'], nesterov = True)
    trainset, devset, testset = datasets
    train_loader = DataLoader(trainset, config['batch_size'], shuffle=True)
    dev_loader = DataLoader(devset, config['batch_size'], shuffle=True)
    #loss_log is episode, step
    #accuracy_log is accuracy
    loss_log, accuracy_log, best_parameters = train(model = model, 
          epochs = config['train_epoch'], 
          train_loader = train_loader,
          dev_loader= dev_loader,
          optimizer= optim,
          lr_scheduler=None,
          device = config['device']
    )
    log_json = {}
    #assuming episode is zero indexed
    for episode, loss in loss_log.items():
       log_json[episode] = dict(
          losses = loss, 
          val_accuracy = accuracy_log[episode]
       )
    log_json["config"] = config
    try:
        with open('baseline_log.json','w') as f:
            json.dump(log_json,f)
        model.save_pretrained("baseline_last")
        model.save_pretrained(save_directory = "baseline_best", state_dict = best_parameters)
    except Exception as e:
       raise e
    return model
    
def jtt(config, datasets):
    model = AutoModelForSequenceClassification.from_pretrained(
      'CAMeL-Lab/bert-base-arabic-camelbert-mix', num_labels=6, id2label=id2label, label2id=label2id
    )
    device = config['device']
    second_model = copy.deepcopy(model)
    if config['optimizer']=="Adam":
       optim = Adam(model.parameters(),config['lr'])
    else:
       optim = SGD(model.parameters(), config['lr'], nesterov = True)
    trainset, devset, testset = datasets
    train_loader = DataLoader(trainset, config['batch_size'], shuffle=True)
    dev_loader = DataLoader(devset, config['batch_size'], shuffle=True)
    #loss_log is episode, step
    #accuracy_log is accuracy
    _ = train(
          model = model, 
          epochs = config['train_epoch'], 
          train_loader = train_loader,
          dev_loader= dev_loader,
          optimizer= optim,
          lr_scheduler=None,
          device = config['device']
    )
    error_indices = generate_upsample_indices(model, train_loader, device)
    print("We will upsampled {} of error data points by {} times".format(len(error_indices), config['E']))
    upsampled_loader = create_upsample_dataloader(trainset, config['batch_size'], error_indices, config['E'])
    if config['optimizer']=="Adam":
       optim = Adam(second_model.parameters(),config['lr'])
    else:
       optim = SGD(second_model.parameters(), config['lr'], nesterov = True)
    loss_log, accuracy_log, best_parameters = train(
          model = second_model, 
          epochs = config['train_epoch'], 
          train_loader = upsampled_loader,
          dev_loader= dev_loader,
          optimizer= optim,
          lr_scheduler=None,
          device = config['device']
    )
    log_json = {}
    #assuming episode is zero indexed
    for episode, loss in loss_log.items():
       log_json[episode] = dict(
          losses = loss, 
          val_accuracy = accuracy_log[episode]
       )
    log_json["config"] = config
    try:
        with open('jtt_log.json','w') as f:
            json.dump(log_json,f)
        #save_model(second_model,'jtt_last.pt')
        #torch.save(best_parameters, 'jtt_best.pt')
        model.save_pretrained("jtt_last")
        model.save_pretrained(save_directory = "jtt_best", state_dict = best_parameters)
    except Exception as e:
       raise e
    
def spare(config,datasets):
    model = AutoModelForSequenceClassification.from_pretrained(
      'CAMeL-Lab/bert-base-arabic-camelbert-mix', num_labels=6, id2label=id2label, label2id=label2id
    )
    second_model = copy.deepcopy(model)
    if config['optimizer']=="Adam":
       optim = Adam(model.parameters(),config['lr'])
    else:
       optim = SGD(model.parameters(), config['lr'], nesterov = True)
    trainset, devset, testset = datasets
    train_loader = DataLoader(trainset, config['batch_size'], shuffle=True)
    dev_loader = DataLoader(devset, config['batch_size'], shuffle=True)
    #loss_log is episode, step
    #accuracy_log is accuracy
    _ = train(model = model, 
          epochs = config['cluster_epoch'], 
          train_loader = train_loader,
          dev_loader= dev_loader,
          optimizer= optim,
          lr_scheduler=None,
          device = config['device']
    )

    device = config['device']
    spare_dataset = copy.deepcopy(trainset)
    #once we copy spare, we need to create a dataloader that loads it in order?
    spare_loader = DataLoader(spare_dataset, batch_size = config['batch_size'], shuffle = False)
    model.eval()
    Z = None
    Labels = []
    Indices = []
    for batch, index in tqdm(spare_loader):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            z = model(input_ids,attention_mask = attention_mask, labels = labels).logits
            if Z is None:
                Z = z.detach().cpu()
            else:
                Z = torch.cat((Z,z.detach().cpu()),dim=0)
            Labels +=labels.detach().cpu().tolist()
            Indices+=index.detach().cpu().tolist()
    inferer = SpareInference(Z= Z, class_labels = Labels, cluster_alg= ClusterAlg.KMEANS, max_clusters = config['max_cluster'], device = device, verbose = False)
    groups, factors= inferer.infer_groups()
    sampling_weights = []
    for key in groups.keys():
        sampling_weights.extend([1 / len(groups[key]) ** factors[key[0]]] * len(groups[key])) #For each cluster,  assign the weights by 1/len(cluster)**lambda
    sampling_weights = np.array(sampling_weights)
    
    class_partition = convert_labels_to_partition(trainset.labels)
    for key in class_partition.keys():
        indices = [x for x in range(len(trainset)) if x in class_partition[key]]
        indices = np.array(indices)
        # normalize the sampling weights so that each class has the same total weight
        sampling_weights[indices] = sampling_weights[indices] / sum(sampling_weights[indices])
    sampling_weights = list(sampling_weights)



    def spare_train(model,epochs,train_set, batch_size,sampling_weights, dev_loader,optimizer,lr_scheduler, device):
        model.to(device)
        loss_log = {}
        accuracy_log = []
        max_accuracy = 0
        best_parameters = None
        for epoch in range(epochs):
            '''Need to update train_loader for every epoch'''
            sampled_indices = random.choices(
               population = list(range(len(train_set))),
               weights = sampling_weights,
               k = len(train_set)
            )
            train_loader = DataLoader(train_set, batch_size, False, CustomIndicesSampler(sampled_indices))
            epoch_loss = []
            model.train()
            print("Start Training:")
            with tqdm(train_loader, unit="batch") as tepoch:
                for batch,index in tepoch:
                    optimizer.zero_grad()
                    tepoch.set_description(f"Epoch {epoch}")
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(input_ids,attention_mask = attention_mask, labels = labels)
                    loss = outputs[0]
                    loss.backward()
                    optimizer.step()
                    if lr_scheduler:
                        lr_scheduler.step()
                    tepoch.set_postfix(loss=loss.item())
                    epoch_loss.append(loss.item())
                loss_log[epoch] = epoch_loss
            model.eval()
            print("Evaluation:")
            num_right = 0
            num_items = 0
            with tqdm(dev_loader, unit="batch") as depoch:
                for batch,index in depoch:
                    depoch.set_description(f"Epoch {epoch}")
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    with torch.no_grad():
                        output = model(input_ids,attention_mask)
                        logits = output.logits
                        predictions = torch.argmax(logits, dim = -1)
                        correct_num = (predictions == labels).sum()
                        num_right += correct_num
                        num_items += len(batch['labels'])
            accuracy = num_right / num_items
            print("accuracy= %.3f" %(accuracy))
            if accuracy.item() > max_accuracy:
                best_parameters = model.state_dict()
                max_accuracy = accuracy.item()
            accuracy_log.append(accuracy.item())
        return loss_log, accuracy_log, best_parameters

    if config['optimizer']=="Adam":
       optim = Adam(second_model.parameters(),config['lr'])
    else:
       optim = SGD(second_model.parameters(), config['lr'], nesterov = True)
    loss_log, accuracy_log, best_parameters = spare_train(second_model, 
                                                          config['train_epoch'], 
                                                          trainset, 
                                                          config['batch_size'], 
                                                          sampling_weights, 
                                                          dev_loader, optim, 
                                                          None, config['device'])

    log_json = {}
    #assuming episode is zero indexed
    for episode, loss in loss_log.items():
       log_json[episode] = dict(
          losses = loss, 
          val_accuracy = accuracy_log[episode]
       )
    log_json["config"] = config
    try:
        with open('spare_log.json','w') as f:
            json.dump(log_json,f)
        model.save_pretrained("spare_last")
        model.save_pretrained(save_directory = "spare_best", state_dict = best_parameters)
    except Exception as e:
       raise e
    return model
   
def dispatch(config, datasets):
    if config['method'] == 'JTT':
       print("Begin JTT")
       jtt(config, datasets)
    elif config['method'] == 'SPARE':
       print("Begin SPARE")
       spare(config, datasets)
    else:
       print("Begin Baseline")
       baseline(config, datasets)


if __name__ == "__main__":
   #Read the Data
   df = pd.read_csv('./full_cleaned_data.tsv',sep='\t')
   tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix")
   id2label = {0: "MSA",1: "MGH",2: "EGY",3: "LEV",4: "IRQ",5: "GLF"}
   label2id = {"MSA":0,"MGH":1,"EGY":2,"LEV":3,"IRQ":4,"GLF":5}
   unique_countries = df['country'].unique()
   country2id = {c:i for i,c in enumerate(list(unique_countries))}
   id2country = {i:c for i,c in enumerate(list(unique_countries))}
   datasets = create_datasets(df, tokenizer,label2id,country2id)
   config = dict(
      method = "SPARE",
      lr = 1e-5,
      optimizer = "Adam",
      train_epoch = 5,
      batch_size = 128,
      device = "cuda",
      E = 5,
      lamb = 2,
      cluster_epoch = 1,
      max_cluster = 10
   )
   dispatch(config, datasets)

