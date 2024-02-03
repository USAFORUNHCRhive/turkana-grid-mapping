# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" This script supports the streaming of large geospatial datasets."""

# Import libraries
import numpy as np
import rasterio
import rasterio.mask
import torch
from rasterio.errors import RasterioIOError
from torch.utils.data.dataset import IterableDataset


class StreamingGeospatialDataset(IterableDataset):
    def __init__(
        self,
        imagery_fns,
        label_fns=None,
        chip_size=256,
        num_chips_per_tile=200,
        kernel_size=20,
        input_channels=4,
        image_transform=None,
        image_augmentation=None,
        label_transform=None,
        nodata_check=None,
        verbose=False,
    ):
        """A torch Dataset for randomly sampling chips from a list of tiles. When used in conjunction with a DataLoader that has `num_workers>1` this Dataset will assign each worker to sample chips from disjoint sets of tiles.
        Args:
            imagery_fns: A list of filenames (or URLS -- anything that `rasterio.open()` can read) pointing to imagery tiles.
            label_fns: A list of filenames of the same size as `imagery_fns` pointing to label mask tiles or `None` if the Dataset should operate in "imagery only mode". Note that we expect `imagery_fns[i]` and `label_fns[i]` to have the same dimension and coordinate system.
            chip_size: Desired size of chips (in pixels).
            num_chips_per_tile: Desired number of chips to sample for each tile.
            kernel_size: Filter size to use for the segmentation filter.
            input_channels: Number of input channels in the dataset.
            image_transform: A function to apply to each image chip object. If this is `None`, then the only transformation applied to the loaded imagery will be to convert it to a `torch.Tensor`. If this is not `None`, then the function should return a `Torch.tensor`.
            image_augmentation: Augmentations to apply to the image chips.
            label_transform: Similar to image_transform, but applied to label chips.
            nodata_check: A method that will check an `(image_chip)` or `(image_chip, label_chip)` (if `label_fns` are provided) and return whether or not the chip should be skipped. This can be used, for example, to skip chips that contain nodata values.
            verbose: If `False` we will be quiet.
        """

        self.fns = list(zip(imagery_fns, label_fns))

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile

        self.image_transform = image_transform
        self.augmentations = image_augmentation
        self.label_transform = label_transform
        self.nodata_check = nodata_check
        self.kernel_size = kernel_size
        self.input_channels = input_channels

        self.verbose = verbose

        if self.verbose:
            print("Constructed StreamingGeospatialDataset")

    def stream_tile_fns(self):
        worker_info = torch.utils.data.get_worker_info()
        if (
            worker_info is None
        ):  # In this case we are not loading through a DataLoader with multiple workers
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # We only want to shuffle the order we traverse the files if we are the first worker (else, every worker will shuffle the files...)
        if worker_id == 0:
            np.random.shuffle(self.fns)  # in place
        # NOTE: A warning, when different workers are created they will all have the same numpy random seed, however will have different torch random seeds. If you want to use numpy random functions, seed appropriately.

        if self.verbose:
            print("Creating a filename stream for worker %d" % (worker_id))

        # This logic splits up the list of filenames into `num_workers` chunks. Each worker will recieve ceil(num_filenames / num_workers) filenames to generate chips from. If the number of workers doesn't divide the number of filenames evenly then the last worker will have fewer filenames.
        N = len(self.fns)
        num_files_per_worker = int(np.ceil(N / num_workers))
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(N, (worker_id + 1) * num_files_per_worker)
        for idx in range(lower_idx, upper_idx):
            label_fn = None
            img_fn, label_fn = self.fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn)

    def stream_chips(self):
        for img_fn, label_fn in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            label_fp = rasterio.open(label_fn, "r")

            height, width = img_fp.shape
            t_height, t_width = label_fp.shape
            assert height == t_height and width == t_width

            try:
                img_data = np.rollaxis(img_fp.read()[: self.input_channels], 0, 3)

                label_data = (
                    label_fp.read().squeeze()
                )  # assume the label geotiff has a single channel
                i = 0
                while i < self.num_chips_per_tile:
                    # Select the top left pixel of our chip randomly
                    x = np.random.randint(0, width - self.chip_size)
                    y = np.random.randint(0, height - self.chip_size)

                    # Read imagery / labels
                    img = None
                    labels = None
                    img = img_data[y : y + self.chip_size, x : x + self.chip_size, :]

                    labels = label_data[y : y + self.chip_size, x : x + self.chip_size]

                    # Check for no data
                    if self.nodata_check is not None:
                        skip_chip = False

                        skip_chip = self.nodata_check(labels)

                        if (
                            skip_chip
                        ):  # The current chip has been identified as invalid by the `nodata_check(...)` method
                            num_skipped_chips += 1
                            continue

                    sample = {
                        "image": np.rollaxis(img, 2, 0)[: self.input_channels, :, :],
                        "mask": labels,
                    }

                    # Apply transform
                    if self.image_transform is not None:
                        sample = self.image_transform(sample)

                    # apply augmentation
                    if self.augmentations is not None:
                        sample["mask"] = (
                            torch.from_numpy(sample["mask"])
                            .type(torch.FloatTensor)
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                        sample["image"] = sample["image"].unsqueeze(0)
                        sample["image"], sample["mask"] = self.augmentations(
                            sample["image"].float(), sample["mask"]
                        )
                        sample["mask"] = sample["mask"].squeeze().long()
                        sample["image"] = sample["image"].squeeze()

                    tmp_labels = sample["mask"].numpy()

                    # create the final labels such that predictions are made on a patch basis
                    final_labels = np.zeros(
                        (
                            tmp_labels.shape[0] // self.kernel_size,
                            tmp_labels.shape[1] // self.kernel_size,
                        )
                    )
                    for ic in range(0, tmp_labels.shape[0], self.kernel_size):
                        for jc in range(0, tmp_labels.shape[1], self.kernel_size):
                            # check that a small block contains a line
                            if (
                                np.sum(
                                    tmp_labels[
                                        ic : ic + self.kernel_size,
                                        jc : jc + self.kernel_size,
                                    ]
                                )
                                > 0
                            ):
                                cur_pos_y, cur_pos_x = (
                                    ic // self.kernel_size,
                                    jc // self.kernel_size,
                                )
                                final_labels[cur_pos_y, cur_pos_x] = 1

                    sample["mask"] = torch.from_numpy(final_labels.astype("int32")).to(
                        torch.long
                    )
                    sample["image"] = sample["image"].type(torch.FloatTensor)
                    sample["mask"] = sample["mask"].unsqueeze(0)

                    i += 1

                    yield sample
            except RasterioIOError as e:
                print("WARNING: Reading %s failed, skipping..." % (img_fn))
                print(f"Error is: {e}")

            # Close file pointers
            img_fp.close()
            label_fp.close()

            if num_skipped_chips > 0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())
