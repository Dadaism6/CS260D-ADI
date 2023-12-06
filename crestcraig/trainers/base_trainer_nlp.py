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
            val_loader,
            test_dataset,
            train_weights: torch.Tensor = None
    ):
        super().__init__(args, model, train_dataset, val_loader, test_dataset,train_weights)
        self.best_val = 0

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
            if self.args.use_wandb:
                wandb.log({
                    "steps": self.steps_per_epoch * epoch + batch_idx,
                    "train_loss_running": loss.item(),
                    "train_acc_running": train_acc})

    def train(self):
        """
        Train the model
        """

        # load checkpoint if resume is True
        if self.args.resume_from_epoch > 0:
            self._load_checkpoint(self.args.resume_from_epoch)

        for epoch in range(self.args.resume_from_epoch, self.args.epochs):
            self._train_epoch(epoch)
            self._val_epoch(epoch)

            self._log_epoch(epoch)

            if self.args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": self.train_loss.val,
                        "train_loss_mean": self.train_loss.avg,
                        "train_acc": self.train_acc.val,
                        "train_acc_mean": self.train_acc.avg,
                        "val_loss": self.val_loss,
                        "val_acc": self.val_acc,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })

            self.lr_scheduler.step()
            if self.val_acc > self.best_val:
                self._save_checkpoint(best=True)
                self.best_val = self.val_acc
            if (epoch + 1) % self.args.save_freq == 0:
                self._save_checkpoint(epoch, best=False)
        self._save_checkpoint()
        self.test_get_group_accuracy(model=self.model, dataset=self.test_dataset, batch_size=self.args.batch_size, device=self.args.device)
    def _save_checkpoint(self, epoch=None, best=False):
        if best:
            save_path = self.args.save_dir + "/model_epoch_best.pt"
        elif epoch is not None:
            save_path = self.args.save_dir + "/model_epoch_{}.pt".format(epoch)
        else:
            save_path = self.args.save_dir + "/model_final.pt"
        self.model.save_pretrained(save_path)
        self.args.logger.info("Checkpoint saved to {}".format(save_path))
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
    # def _test_epoch(self, epoch):
    #     save_path = self.args.save_dir + "/model_epoch_best.pt"
    #     self.model.from_pretrained(save_path)
    #     self.model.eval()
    #
    #     test_loss = 0
    #     test_acc = 0
    #
    #     with torch.no_grad():
    #         for batch in self.test_loader:
    #             # Move data to the device
    #             input_ids = batch['input_ids'].to(self.args.device)
    #             attention_mask = batch['attention_mask'].to(self.args.device)
    #             labels = batch['label'].to(self.args.device)
    #
    #             # Forward pass
    #             outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #
    #             # Accumulate loss and accuracy
    #             loss = outputs.loss
    #             test_loss += loss.item() * input_ids.size(0)
    #             preds = outputs.logits.argmax(dim=-1)
    #             test_acc += (preds == labels).float().sum().item()
    #
    #     # Calculate average loss and accuracy over all of the dataset
    #     test_loss /= len(self.test_loader.dataset)
    #     test_acc /= len(self.test_loader.dataset)
    #
    #     self.test_loss = test_loss
    #     self.test_acc = test_acc
    #     print("Final test loss: {:.4f}, Final test acc: {:.4f}".format(test_loss, test_acc))
    #     if self.args.use_wandb:
    #         wandb.log(
    #             {
    #                 "test_loss": test_loss,
    #                 "test_acc": test_acc,
    #             })

    def get_accuracy_no_tqdm(self, model, loader, device):
        num_items = 0
        num_correct = 0
        model.eval()
        step = 0
        with torch.no_grad():
            for batch, index in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                right = (predictions == labels).sum()
                num = len(input_ids)
                num_items += num
                num_correct += right
                step += 1
        return (num_correct / num_items).item()

    def test_get_group_accuracy(self, model, dataset, batch_size, device):
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
            accuracies[key] = self.get_accuracy_no_tqdm(model, testloaders[key], device)
            print(f"Group {key} Accuracy: {accuracies[key]}")
        save_path = self.args.save_dir + "/group_accuracy.json"
        with open(save_path, "w") as f:
            json.dump(accuracies, f)