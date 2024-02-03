# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This script supports streaming data from a set of tiles given pole locations.
"""
# Import required libraries
import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.mask
import torch
from rasterio.errors import RasterioIOError
from shapely.geometry import Point, box
from torch.utils.data.dataset import IterableDataset


class StreamingGeospatialDataset(IterableDataset):
    def __init__(
        self,
        imagery_fns,
        label_fns=None,
        data_mask_fns=None,
        chip_size=256,
        num_chips_per_tile=200,
        input_channels=4,
        vector_labels_fn=None,
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
            data_mask_fns: A list of filenames of the same size as `imagery_fns` pointing to valid data tiles or `None` if the Dataset should operate in "imagery only mode". Note that we expect `imagery_fns[i]` and `label_fns[i]` to have the same dimension and coordinate system.
            chip_size: Desired size of chips (in pixels).
            num_chips_per_tile: Desired number of chips to sample for each tile.
            input_channels: Desired number of input channels.
            vector_labels_fn: A filename pointing to a vector file containing labels. If this is not `None` then `label_fns` must be `None`.
            image_transform: A function to apply to each image chip object. If this is `None`, then the only transformation applied to the loaded imagery will be to convert it to a `torch.Tensor`. If this is not `None`, then the function should return a `Torch.tensor`. Further, if `groups` is not `None` then the transform function should expect the imagery as the first argument and the group as the second argument.
            label_transform: Similar to image_transform, but applied to label chips.
            nodata_check: A method that will check an `(image_chip)` or `(image_chip, label_chip)` (if `label_fns` are provided) and return whether or not the chip should be skipped. This can be used, for example, to skip chips that contain nodata values.
            verbose: If `False` we will be quiet.
        """

        if label_fns is None or vector_labels_fn is not None:
            self.fns = imagery_fns
            self.use_labels = False
        else:
            self.fns = list(zip(imagery_fns, label_fns, data_mask_fns))
            self.use_labels = True

        self.vector_labels_fn = vector_labels_fn

        self.chip_size = chip_size
        self.num_chips_per_tile = num_chips_per_tile

        self.image_transform = image_transform
        self.augmentations = image_augmentation
        self.label_transform = label_transform
        self.nodata_check = nodata_check
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
            data_mask_fn = None
            if self.use_labels:
                img_fn, label_fn, data_mask_fn = self.fns[idx]
            else:
                img_fn = self.fns[idx]

            if self.verbose:
                print("Worker %d, yielding file %d" % (worker_id, idx))

            yield (img_fn, label_fn, data_mask_fn)

    def stream_chips(self):
        for img_fn, label_fn, data_mask_fn in self.stream_tile_fns():
            num_skipped_chips = 0

            # Open file pointers
            img_fp = rasterio.open(img_fn, "r")
            img_crs = img_fp.crs
            label_fp = rasterio.open(label_fn, "r") if self.use_labels else None
            valid_data_mask_fp = (
                rasterio.open(data_mask_fn, "r") if self.use_labels else None
            )

            height, width = img_fp.shape
            if (
                self.use_labels
            ):  # garantee that our label mask has the same dimensions as our imagery
                t_height, t_width = label_fp.shape
                assert height == t_height and width == t_width

            try:
                # If we aren't in windowed sampling mode then we should read the entire tile up front
                img_data = np.rollaxis(img_fp.read()[: self.input_channels], 0, 3)
                if self.use_labels:
                    label_data = (
                        label_fp.read().squeeze()
                    )  # assume the label geotiff has a single channel
                    data_mask = valid_data_mask_fp.read().squeeze()
                elif self.vector_labels_fn is not None:
                    with fiona.open(self.vector_labels_fn, "r") as shapefile:
                        poly_crs = shapefile.crs
                        shapes = [feature["geometry"] for feature in shapefile]
                        types = [feature["properties"]["type"] for feature in shapefile]
                        bounds = img_fp.bounds
                        geom = box(*bounds)

                        matches_shapes = []
                        matches_types = []
                        for idx, shape in enumerate(shapes):
                            warped_shape = fiona.transform.transform(
                                poly_crs,
                                img_crs,
                                [shape["coordinates"][0]],
                                [shape["coordinates"][1]],
                            )
                            pt = Point(warped_shape)
                            if geom.contains(pt):
                                matches_shapes.append(warped_shape)
                                matches_types.append(types[idx])

                        label_data = np.zeros(
                            (img_fp.read().shape[1], img_fp.read().shape[2])
                        )
                        data_mask = np.zeros(
                            (img_fp.read().shape[1], img_fp.read().shape[2])
                        )
                        for i, (lon, lat) in enumerate(matches_shapes):
                            py, px = img_fp.index(lon, lat)
                            data_mask[py, px] = 1
                            if matches_types[i] == "pole":
                                label_data[py, px] = 1

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
                    mask = data_mask[y : y + self.chip_size, x : x + self.chip_size]

                    # Check for no data using the mask (i.e. only samples from either pole or hard negative location)
                    if self.nodata_check is not None:
                        skip_chip = False
                        skip_chip = self.nodata_check(mask)

                        if (
                            skip_chip
                        ):  # The current chip has been identified as invalid by the `nodata_check(...)` method
                            num_skipped_chips += 1
                            continue

                    sample = {"image": np.rollaxis(img, 2, 0), "mask": labels}

                    # Transform the imagery
                    if self.image_transform is not None:
                        self.image_transform(sample)
                    else:
                        sample["image"] = torch.from_numpy(img)

                    # Data augmentation
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
                        sample["mask"] = sample["mask"].squeeze(0).long()
                        sample["image"] = sample["image"].squeeze(0)

                    i += 1
                    yield sample
            except (
                RasterioIOError
            ) as e:  # NOTE: To catch weird errors that I was seeing occasionally when trying to read from COGS - I don't remember the details though
                print("WARNING: Reading %s failed, skipping..." % (img_fn))
                print(f"Error is: {e}")

            # Close file pointers
            img_fp.close()
            if self.use_labels:
                label_fp.close()

            if num_skipped_chips > 0 and self.verbose:
                print("We skipped %d chips on %s" % (img_fn))

    def __iter__(self):
        if self.verbose:
            print("Creating a new StreamingGeospatialDataset iterator")
        return iter(self.stream_chips())
