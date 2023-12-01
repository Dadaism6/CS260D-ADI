import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
# from utils import GradualWarmupScheduler
from transformers import AutoModelForSequenceClassification, AdamW
from crestcraig.trainers.base_trainer import BaseTrainer
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
            self, val, n=1
    ):  # n is the number of samples in the batch, default to 1
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NLPBaseTrainer(BaseTrainer):
    def __init__(
            self,
            args: argparse.Namespace,
            model: nn.Module,
            train_dataset,
            val_dataset,
            train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_dataset, train_weights)

    def _forward_and_backward(self, batch):
        self.optimizer.zero_grad()

        # Unpack the batch data
        input_ids = batch['input_ids'].to(self.args.device)
        attention_mask = batch['attention_mask'].to(self.args.device)
        labels = batch['label'].to(self.args.device)
        data_idx = batch['index']

        # Forward pass
        forward_start = time.time()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        forward_time = time.time() - forward_start
        self.batch_forward_time.update(forward_time)

        # Extract the loss
        loss = outputs.loss
        loss = (loss * self.train_weights[data_idx]).mean()

        # Backward pass
        backward_start = time.time()
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.batch_backward_time.update(backward_time)

        # Compute accuracy
        preds = outputs.logits.argmax(dim=-1)
        train_acc = (preds == labels).float().mean().item()

        # Update training loss and accuracy
        self.train_loss.update(loss.item(), input_ids.size(0))
        self.train_acc.update(train_acc, input_ids.size(0))

        return loss, train_acc

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout)
        for batch_idx, batch in pbar:
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            loss, train_acc = self._forward_and_backward(batch)

            pbar.set_description("Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                epoch,
                self.args.epochs,
                batch_idx * self.args.batch_size + len(batch['input_ids']),
                len(self.train_loader.dataset),
                100.0 * (batch_idx + 1) / len(self.train_loader),
                loss.item(),
                train_acc,
            ))

            data_start = time.time()

    def _val_epoch(self, epoch):
        self.model.eval()

        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to the device
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                labels = batch['label'].to(self.args.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Accumulate loss and accuracy
                loss = outputs.loss
                val_loss += loss.item() * input_ids.size(0)
                preds = outputs.logits.argmax(dim=-1)
                val_acc += (preds == labels).float().sum().item()

        # Calculate average loss and accuracy over all of the dataset
        val_loss /= len(self.val_loader.dataset)
        val_acc /= len(self.val_loader.dataset)

        self.val_loss = val_loss
        self.val_acc = val_acc