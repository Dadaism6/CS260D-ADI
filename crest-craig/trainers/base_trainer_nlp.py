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
from base_trainer import BaseTrainer
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
            tokenizer: BertTokenizer,
            train_dataset,
            val_dataset,
            train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_dataset, train_weights)
        self.tokenizer = tokenizer

    def _prepare_batch(self, batch):
        input_ids = batch[0]['input_ids'].squeeze(1).to(self.args.device)
        attention_mask = batch[0]['attention_mask'].squeeze(1).to(self.args.device)
        labels = batch[1].to(self.args.device)
        return input_ids, attention_mask, labels

    def _forward_and_backward(self, data, target, data_idx):
        input_ids, attention_mask, labels = self._prepare_batch((data, target))

        self.optimizer.zero_grad()
        forward_start = time.time()
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        forward_time = time.time() - forward_start
        self.batch_forward_time.update(forward_time)

        loss = outputs.loss
        backward_start = time.time()
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.batch_backward_time.update(backward_time)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()

        self.train_loss.update(loss.item(), input_ids.size(0))
        self.train_acc.update(acc, input_ids.size(0))

        return loss, acc

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        pbar = tqdm(self.train_loader, total=len(self.train_loader), file=sys.stdout)
        for batch in pbar:
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            loss, train_acc = self._forward_and_backward(batch)

            pbar.set_description(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    self.args.epochs,
                    pbar.n * self.args.batch_size,
                    len(self.train_loader.dataset),
                    100.0 * pbar.n / len(self.train_loader),
                    loss.item(),
                    train_acc,
                )
            )

            data_start = time.time()

    def _val_epoch(self, epoch):
        self.model.eval()
        self.val_loss.reset()
        self.val_acc.reset()

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = self._prepare_batch(batch)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean().item()

                self.val_loss.update(loss.item(), inputs["input_ids"].size(0))
                self.val_acc.update(acc, inputs["input_ids"].size(0))

        # Logging validation loss and accuracy
        self.args.logger.info(
            "Validation Epoch {}:\tVal Loss: {:.6f}\tVal Acc: {:.6f}".format(
                epoch, self.val_loss.avg, self.val_acc.avg
            )
        )