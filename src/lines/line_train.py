# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This script is used to train the line segmentation model. Some examples are provided below based on the config.yaml file.
"""

# Import libraries
import argparse
import os

import lightning.pytorch as pl
from data.dataloader_line import LineDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lineseg.line_tasks import SemanticSegmentationTask
from omegaconf import OmegaConf


def main(config):
    """
    Main training routine specific for line segmentation using the config file."""
    datamodule = LineDataModule(
        **config.datamodule,
        num_classes=config["learning"]["num_classes"],
        input_channels=config["learning"]["in_channels"],
    )

    task = SemanticSegmentationTask(**config.learning)
    ckpt_dir_path = (
        f"checkpoints/{config['evaluation']['method_name'].lower().replace(' ', '_')}"
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=config["learning"]["early_stopping_patience"],
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=ckpt_dir_path,
        filename="best_model",
        save_top_k=1,
        mode="min",
    )
    logger = pl.loggers.TensorBoardLogger(
        "logs/",
        name=f"{config['evaluation']['method_name'].lower().replace(' ', '_')}",
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        logger=logger,
        **config.trainer,
    )
    trainer.fit(model=task, datamodule=datamodule)
    OmegaConf.save(config, os.path.join(ckpt_dir_path, "config.yaml"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method_name",
        required=True,
        type=str,
        help="The name of the line segmentation experiment.",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="line_config.yaml",
        help="Path to config.yaml file",
    )

    args = parser.parse_args()
    config = OmegaConf.load(args.config_file_path)
    config["evaluation"]["method_name"] = args.method_name
    config["trainer"]["devices"] = [args.gpu] if args.gpu != -1 else -1
    main(config)
