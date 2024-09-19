# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pole detection tasks. This utilizes task framework similar to the torchgeo segmentation tasks: 
https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/trainers/segmentation.html#SemanticSegmentationTask"""
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import numpy as np
import poledetect.lcfcn_loss as lcfcn_loss
import torch
from lightning.pytorch import LightningModule
from poledetect.fcn8_resnet import FCN8
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchgeo.datasets.utils import unbind_samples
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError


class SemanticSegmentationTask(LightningModule):
    """LightningModule for point segmentation of images."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""

        if self.hyperparams["model"] == "fcn8":
            self.model = FCN8()
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['model']}' is not valid. "
                "Currently, only supports 'lcfcn'."
            )

        if self.hyperparams["loss"] == "lcfcn":
            self.loss = lcfcn_loss.compute_loss
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'lcfcn' loss."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            model: Name of the segmentation model type to use
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function, currently supports
                lcfcn loss
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

        self.train_metrics = MetricCollection(
            [
                MeanAbsoluteError(),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)

        loss = 0
        for i in range(y_hat.shape[0]):
            loss += self.loss(y[i], y_hat[i].sigmoid()).nanmean()
        loss = loss / y_hat.shape[0]

        probs = y_hat.sigmoid().detach().cpu().numpy()

        pred_count = []

        for i in range(y_hat.shape[0]):
            blobs = lcfcn_loss.get_blobs(probs[i, 0, :, :])
            pred_count.append(float((np.unique(blobs) != 0).sum()))

        pred_count = torch.tensor(pred_count)
        true_count = ((y) != 0).sum(axis=[1, 2]).detach().cpu().type(torch.FloatTensor)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.train_metrics(pred_count, true_count)

        return cast(Tensor, loss)

    def on_train_epoch_end(self) -> None:
        """Logs epoch level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)

        loss = 0
        for i in range(y_hat.shape[0]):
            loss += self.loss(y[i], y_hat[i].sigmoid()).nanmean()
        loss = loss / y_hat.shape[0]
        self.log("val_loss", loss, on_step=False, on_epoch=True,prog_bar=True)

        y_hat_hard = y_hat.sigmoid()
        probs = y_hat.sigmoid().detach().cpu().numpy()

        pred_count = []

        for i in range(y_hat.shape[0]):
            blobs = lcfcn_loss.get_blobs(probs[i, 0, :, :])
            pred_count.append(float((np.unique(blobs) != 0).sum()))

        pred_count = torch.tensor(pred_count)
        true_count = ((y) != 0).sum(axis=[1, 2]).detach().cpu().type(torch.FloatTensor)
        self.val_metrics(pred_count, true_count)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()
            except ValueError:
                pass

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)

        loss = 0
        for i in range(y_hat.shape[0]):
            loss += self.loss(y[i], y_hat[i].sigmoid()).nanmean()
        loss = loss / y_hat.shape[0]

        probs = y_hat.sigmoid().detach().cpu().numpy()

        pred_count = []

        for i in range(y_hat.shape[0]):
            blobs = lcfcn_loss.get_blobs(probs[i, 0, :, :])
            pred_count.append(float((np.unique(blobs) != 0).sum()))

        pred_count = torch.tensor(pred_count)
        true_count = ((y) != 0).sum(axis=[1, 2]).detach().cpu().type(torch.FloatTensor)

        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(pred_count, true_count)

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the predictions.

        By default, this will loop over images in a dataloader and aggregate
        predictions into a list. This may not be desirable if you have many images
        or large images which could cause out of memory errors. In this case
        it's recommended to override this with a custom predict_step.

        Args:
            batch: the output of your DataLoader

        Returns:
            predicted softmax probabilities
        """
        batch = args[0]
        x = batch["image"]
        y_hat: Tensor = self(x).softmax(dim=1)
        return y_hat

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            learning rate dictionary
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }
