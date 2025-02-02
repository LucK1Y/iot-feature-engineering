# To Avoid Crashes with a lot of nodes
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

import lightning as pl
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from torchmetrics import MetricCollection


class MalwaresModelLinearLayer(pl.LightningModule):
    def process_metrics(self, phase, y_pred, y, loss=None):
        if loss is not None:
            self.log(f"{phase}/Loss", loss, prog_bar=True, logger=True)

        y_pred_classes = torch.argmax(y_pred, dim=1)
        if phase == "Train":
            output = self.train_metrics(y_pred_classes, y)
        elif phase == "Validation":
            output = self.val_metrics(y_pred_classes, y)
        elif phase == "Test":
            output = self.test_metrics(y_pred_classes, y)
        else:
            raise NotImplementedError
        # print(f"y_pred shape: {y_pred.shape}, y_pred_classes shape: {y_pred_classes.shape}, y shape: {y.shape}")  # Debug print
        output = {
            f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value
            for key, value in output.items()
        }
        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            self.cm.update(y_pred_classes, y)

    def log_metrics_by_epoch(self, phase, print_cm=True, plot_cm=True):
        print(f"Epoch end: {phase}, epoch number: {self.epoch_global_number[phase]}")
        if phase == "Train":
            output = self.train_metrics.compute()
            self.train_metrics.reset()
        elif phase == "Validation":
            output = self.val_metrics.compute()
            self.val_metrics.reset()
        elif phase == "Test":
            output = self.test_metrics.compute()
            self.test_metrics.reset()
        else:
            raise NotImplementedError

        output = {
            f"{phase}Epoch/{key.replace('Multiclass', '').split('/')[-1]}": value
            for key, value in output.items()
        }

        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            cm = self.cm.compute().cpu()
            print(f"{phase}Epoch/CM\n", cm) if print_cm else None
            if plot_cm:
                import seaborn as sns
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 7))
                ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
                ax.set_xlabel("Predicted labels")
                ax.set_ylabel("True labels")
                ax.set_title("Confusion Matrix")
                ax.set_xticks(range(9))
                ax.set_yticks(range(9))
                ax.xaxis.set_ticklabels([i for i in range(9)])
                ax.yaxis.set_ticklabels([i for i in range(9)])
                self.logger.experiment.add_figure(
                    f"{phase}Epoch/CM",
                    ax.get_figure(),
                    global_step=self.epoch_global_number[phase],
                )
                plt.close()

        # Reset metrics
        self.epoch_global_number[phase] += 1

    def __init__(
        self,
        input_size=30,
        hidden_size1=30,
        hidden_size2=30,
        output_size=9,
        learning_rate=1e-2,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        super().__init__()
        if metrics is None:
            metrics = MetricCollection(
                [
                    MulticlassAccuracy(num_classes=output_size),
                    MulticlassPrecision(num_classes=output_size),
                    MulticlassRecall(num_classes=output_size),
                    MulticlassF1Score(num_classes=output_size),
                ]
            )

        # Define metrics
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        if confusion_matrix is None:
            self.cm = MulticlassConfusionMatrix(num_classes=output_size)

        # Set seed for reproducibility initialization
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss()

        self.l1 = torch.nn.Linear(input_size, output_size)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def forward(self, x):
        x = self.l1(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, phase):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)

        # Get metrics for each batch and log them
        self.log(f"{phase}/Loss", loss, prog_bar=True)
        self.process_metrics(phase, y_pred, y, loss)

        return loss

    def training_step(self, batch, batch_id):
        return self.step(batch, "Train")

    def on_train_epoch_end(self):
        self.log_metrics_by_epoch("Train", print_cm=True, plot_cm=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "Validation")

    def on_validation_epoch_end(self):
        self.log_metrics_by_epoch("Validation", print_cm=True, plot_cm=True)

    def test_step(self, batch, batch_idx):
        return self.step(batch, "Test")

    def on_test_epoch_end(self):
        self.log_metrics_by_epoch("Test", print_cm=True, plot_cm=True)
