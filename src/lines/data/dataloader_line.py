# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This script contains the dataloader for the line dataset. 
"""

# Import libraries
import glob

import kornia.augmentation as K
import lightning.pytorch as pl
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as T
from data.streaming_line_dataset import StreamingGeospatialDataset
from einops import rearrange
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

# Define the default augmentations
DEFAULT_AUGS = K.AugmentationSequential(
    K.RandomRotation(p=0.5, degrees=90),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.ColorJitter(p=0.5, brightness=0.5, contrast=0.5, saturation=0.5),
    data_keys=["image", "mask"],
)

REFINEMENT_AUGS = K.AugmentationSequential(
    K.RandomRotation(p=0.5, degrees=90),
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    data_keys=["image", "mask"],
)


def get_fns(base_dir, split):
    """Get the filenames for the given split.

    Args:
        base_dir (str): The base directory name containing the split directories.
        split (str): The split to get the image filenames for.

    Returns:
        list: A list of image filenames for the given split.
    """
    filenames = glob.glob(base_dir + split + "/*.tif")
    return sorted(filenames)


def nodata_check(label):
    """Check if the label is all nodata.

    Args:
        label (array): Input label array.

    Returns:
        bool : True if the label is all nodata, False otherwise.
    """
    flag = False
    if np.sum(label) == 0:
        flag = True
    return flag


class Preprocessor:
    """Data preprocessing: normalize and standardize the data."""

    def __init__(self, img_mean, img_std, standardize=False) -> None:
        self.img_min = img_mean
        self.img_max = img_std
        self.standardize = standardize

    def __call__(self, sample):

        # Perform mean-std standardization
        if self.standardize:
            sample["image"] = (sample["image"] - self.img_mean) / self.img_std
        else:
            # RGB is uint8 so divide by 255
            sample["image"] = (
                torch.from_numpy(sample["image"]).type(torch.FloatTensor) / 255.0
            )

        return sample


class LineDataModule(pl.LightningDataModule):
    """Data module for the line dataset."""

    def __init__(
        self,
        root_dir: str,
        chip_size: int = 256,
        num_chips_per_tile: int = 60,
        num_workers: int = 8,
        batch_size: int = 8,
        segm_filter_size: int = 1,
        img_mean: list = [75.006010333223, 68.318348580367, 63.633123192871],
        img_std: list = [76.842842915691, 70.10749190534, 65.514465248706],
        num_classes: int = 2,
        input_channels: int = 4,
        **kwargs,
    ):
        """Initialize the line dataset.
        Args:
            root_dir (str): The root directory containing the data.
            chip_size (int): The size of the chips to sample from the tiles.
            num_chips_per_tile (int): The number of chips to sample from each tile.
            num_workers (int): The number of workers to use for the dataloaders.
            batch_size (int): The batch size to use for the dataloaders.
            segm_filter_size (int): The size of the filter to use for the segmentation filter.
            img_mean (list): The mean pixel values for the dataset.
            img_std (list): The standard deviation of the pixel values for the dataset.
            num_classes (int): The number of classes in the dataset.
            input_channels (int): The number of input channels in the dataset.
        """
        super().__init__()
        self.root_dir = root_dir
        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.segm_filter_size = segm_filter_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.augmentations = DEFAULT_AUGS
        self.preprocess = Preprocessor(self.img_mean, self.img_std)
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage=None):
        """Setup the datasets for the line dataset."""
        transforms = T.Compose([self.preprocess])
        # create the training dataset
        train_img_fns = get_fns(self.root_dir, "train_images")
        train_label_fns = get_fns(self.root_dir, "train_masks_lines")
        self.train_ds = StreamingGeospatialDataset(
            train_img_fns,
            train_label_fns,
            chip_size=self.chip_size,
            num_chips_per_tile=self.num_chips_per_tile,
            image_transform=transforms,
            image_augmentation=self.augmentations,
            verbose=False,
            nodata_check=nodata_check,
            kernel_size=self.segm_filter_size,
            input_channels=self.input_channels,
        )
        # create the validation dataset
        val_img_fns = get_fns(self.root_dir, "val_images")
        val_label_fns = get_fns(self.root_dir, "val_masks_lines")
        self.val_ds = StreamingGeospatialDataset(
            val_img_fns,
            val_label_fns,
            chip_size=self.chip_size,
            num_chips_per_tile=8,
            image_transform=transforms,
            image_augmentation=self.augmentations,
            verbose=False,
            nodata_check=nodata_check,
            kernel_size=self.segm_filter_size,
            input_channels=self.input_channels,
        )

        # create the test dataset
        test_img_fns = get_fns(self.root_dir, "test_images")
        test_label_fns = get_fns(self.root_dir, "test_masks_lines")
        self.test_ds = StreamingGeospatialDataset(
            test_img_fns,
            test_label_fns,
            chip_size=self.chip_size,
            num_chips_per_tile=8,
            image_transform=transforms,
            image_augmentation=None,
            verbose=False,
            nodata_check=nodata_check,
            kernel_size=self.segm_filter_size,
            input_channels=self.input_channels,
        )

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def on_after_batch_transfer(self, batch, dl_idx):
        """Move the mask to the last channel."""
        batch["mask"] = rearrange(batch["mask"], "b () h w -> b h w")
        return batch

    def plot(self, sample):
        """Plot a single sample from the dataset."""
        image = np.rollaxis(sample["image"].numpy(), 0, 3)
        mask = sample["mask"].squeeze(0)
        mask = scipy.ndimage.zoom(mask.numpy(), self.segm_filter_size, order=1)
        ncols = 2
        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            pred = scipy.ndimage.zoom(pred.numpy(), self.segm_filter_size, order=1)
            ncols += 1

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))

        axs[0].imshow(image[:, :, :3])
        axs[0].axis("off")
        axs[1].imshow(mask, vmin=0, vmax=self.num_classes - 1, interpolation="none")
        axs[1].axis("off")

        if showing_predictions:
            axs[2].imshow(pred, vmin=0, vmax=self.num_classes - 1, interpolation="none")
            axs[2].axis("off")

        return fig
