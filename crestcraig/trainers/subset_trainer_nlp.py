from .base_trainer import *
from torch.utils.data import Subset, DataLoader
from crestcraig.datasets.subset import SubsetGenerator
from crestcraig.trainers.subset_trainer import SubsetTrainer
from crestcraig.trainers.base_trainer_nlp import NLPBaseTrainer

class NLPSubsetTrainer(NLPBaseTrainer):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: DataLoader,
        train_weights: torch.Tensor = None,
    ):
        super().__init__(args, model, train_dataset, val_loader, train_weights)
        self.train_target = np.array(self.train_dataset.dataset.label)
        self.subset_generator = SubsetGenerator(greedy=(args.selection_method != "rand"), smtk=args.smtk)

        self.num_selection = 0
    def _train_epoch(self, epoch):
        # select a subset of the data
        self._select_subset(epoch, len(self.train_loader) * epoch)
        self._update_train_loader_and_weights()

        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        # use tqdm to display a smart progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout)
        for batch_idx, batch in pbar:

            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            self.optimizer.zero_grad()

            # train model with the current batch and record forward and backward time
            loss, train_acc = self._forward_and_backward(batch)

            data_start = time.time()

            # update progress bar
            pbar.set_description(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    self.args.epochs,
                    batch_idx * self.args.batch_size,
                    len(self.train_loader.dataset),
                    100.0 * (batch_idx+1) / len(self.train_loader),
                    loss.item(),
                    train_acc,
                )
            )
            if self.args.use_wandb:
                wandb.log({
                    "steps": self.steps_per_epoch * epoch + batch_idx,
                    "train_loss_running": loss.item(),
                    "train_acc_running": train_acc})

        if self.args.cache_dataset and self.args.clean_cache_iteration:
            self.train_dataset.clean()
            self._update_train_loader_and_weights()
        if self.args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_acc": train_acc})

    def _get_train_output(self):
        """
        Evaluate the model on the training set and record the output and softmax
        """
        self.model.eval()

        self.train_output = np.zeros((len(self.train_dataset), self.args.num_classes))
        self.train_softmax = np.zeros((len(self.train_dataset), self.args.num_classes))

        with torch.no_grad():
            for _, batch in enumerate(self.train_val_loader):
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                labels = batch['label'].to(self.args.device)
                data_idx = batch['index']

                output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                self.train_output[data_idx] = output.logits.cpu().numpy()
                self.train_softmax[data_idx] = output.logits.softmax(dim=1).cpu().numpy()

        self.model.train()


    #NOTE: NO CHANGE
    def _update_train_loader_and_weights(self):
        self.args.logger.info("Updating train loader and weights with subset of size {}".format(len(self.subset)))
        self.train_loader = DataLoader(
            Subset(self.train_dataset, self.subset),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.train_val_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.train_weights = np.zeros(len(self.train_dataset))
        self.subset_weights = self.subset_weights / np.sum(self.subset_weights) * len(self.subset)
        self.train_weights[self.subset] = self.subset_weights
        self.train_weights = torch.from_numpy(self.train_weights).float().to(self.args.device)
    def _select_subset(self, epoch: int, training_step: int):
        """
        Select a subset of the data
        """
        self.num_selection += 1

        if self.args.use_wandb:
            wandb.log({"epoch": epoch, "training_step": training_step, "num_selection": self.num_selection})

        if self.args.cache_dataset:
            self.train_dataset.clean()
            self.train_dataset.cache()
        pass

    def _get_num_selection(self):
        return self.num_selection