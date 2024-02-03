# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Line Segmentation tasks. This utilizes task framework similar to the torchgeo segmentation tasks: 
https://torchgeo.readthedocs.io/en/stable/_modules/torchgeo/trainers/segmentation.html#SemanticSegmentationTask"""

import warnings
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchgeo.datasets.utils import unbind_samples
from torchgeo.models import FCN
from torchmetrics import MetricCollection
from torchmetrics.classification import (MulticlassAccuracy,
                                         MulticlassJaccardIndex)

from .models import LineUnetModel, UnetModel


class SemanticSegmentationTask(LightningModule):
    """LightningModule for segmentation of lines from images."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""

        if self.hyperparams["model"] == "line_unet":
            self.model = LineUnetModel(
                in_channels=self.hyperparams["in_channels"],
                num_classes=self.hyperparams["num_classes"],
                stride=self.hyperparams["stride"],
                use_maxpool=self.hyperparams["use_maxpool"],
            )
        elif self.hyperparams["model"] == "unet":
            self.model = UnetModel(
                in_channels=self.hyperparams["in_channels"],
                num_classes=self.hyperparams["num_classes"],
                stride=self.hyperparams["stride"],
                use_maxpool=self.hyperparams["use_maxpool"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['model']}' is not valid. "
                "Currently, only supports 'line_unet' and 'unet'."
            )

        if self.hyperparams["loss"] == "ce":
            ignore_value = -1000 if self.ignore_index is None else self.ignore_index
            self.loss = nn.CrossEntropyLoss(
                ignore_index=ignore_value,
                reduction="none",
                weight=torch.Tensor(self.hyperparams["loss_weights"]),
            )
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports weighted 'ce' loss."
            )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            model: Name of the segmentation model type to use
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            stride: Stride to apply in covolution
            use_maxpool: Use maxpool or average pool
            loss: Name of the loss function, currently supports weighted
                'ce' loss
            ignore_index: Optional integer class index to ignore in the loss and metrics
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        if not isinstance(kwargs["ignore_index"], (int, type(None))):
            raise ValueError("ignore_index must be an int or None")
        if (kwargs["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.ignore_index = kwargs["ignore_index"]
        self.config_task()

        self.train_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    multidim_average="global",
                    average="macro",
                ),
                MulticlassJaccardIndex(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    average="macro",
                ),
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
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y).nanmean()

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

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
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y).nanmean()

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

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
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y).nanmean()

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

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
