# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""""
This script prepares the data for the pole detection task."""
import argparse
import multiprocessing
import os

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.mask
from omegaconf import OmegaConf
from shapely.geometry import Point, box


def create_and_save_mask(
    img_fn, poly_crs, shapes, types, label_save_path, valid_datamask_save_path
):
    """Create and save the mask of points for the given image.

    Args:
        img_fn (str): Input image name
        poly_crs (int): Crs of the pole vector file
        shapes (list): Geometies in the pole vector file
        types (list): Geometry attributes (pole OR not pole) in the pole vector file
        label_save_path (str): Path to save the generated label mask
        valid_datamask_save_path (str): Path to save the generated valid data mask (such that sampling can be done from areas with poles & hard negatives)
    """
    print(f"Working on image: {img_fn}")
    img_fp = rasterio.open(os.path.join(img_fn), "r")
    img_crs = img_fp.crs
    bounds = img_fp.bounds
    geom = box(*bounds)
    matches_shapes = []
    matches_types = []
    # Find line strings that are in the image
    for idx, shape in enumerate(shapes):
        warped_shape = fiona.transform.transform(
            poly_crs, img_crs, [shape["coordinates"][0]], [shape["coordinates"][1]]
        )
        pt = Point(warped_shape)
        if geom.contains(pt):
            matches_shapes.append(warped_shape)
            matches_types.append(types[idx])

        label_data = np.zeros((img_fp.read().shape[1], img_fp.read().shape[2]))
        data_mask = np.zeros((img_fp.read().shape[1], img_fp.read().shape[2]))
        for i, (lon, lat) in enumerate(matches_shapes):
            py, px = img_fp.index(lon, lat)
            data_mask[py, px] = 1
            if matches_types[i] == "pole":
                label_data[py, px] = 1

    output_profile = img_fp.profile.copy()
    output_profile["dtype"] = "uint8"
    output_profile["count"] = 1

    mask_profile = img_fp.profile.copy()
    mask_profile["dtype"] = "uint8"
    mask_profile["count"] = 1
    # Save the generated mask
    token = img_fn.split("/")[-1]
    print(f"Saving : {token}")
    with rasterio.open(
        os.path.join(label_save_path, token), "w", **output_profile
    ) as f:
        f.write(label_data, 1)

    with rasterio.open(
        os.path.join(valid_datamask_save_path, token), "w", **mask_profile
    ) as f:
        f.write(data_mask, 1)


def main(config):
    """This script prepares the data for the line segmentation task."""

    # Get the line features from the geojson file
    with fiona.open(config["dataprep"]["pole_vector_file"], "r") as shapefile:
        poly_crs = shapefile.crs
        shapes = [feature["geometry"] for feature in shapefile]
        types = [feature["properties"]["type"] for feature in shapefile]

    # Loop through the data splits and create the masks
    for split in config["dataprep"]["data_splits"]:
        img_fns = [
            os.path.join(config["datamodule"]["root_dir"], f"{split}_images", k)
            for k in os.listdir(
                os.path.join(config["datamodule"]["root_dir"], f"{split}_images")
            )
            if k.endswith(".tif")
        ]

        label_save_path = os.path.join(
            config["datamodule"]["root_dir"], f"{split}_masks_poles"
        )
        if not os.path.exists(label_save_path):
            os.makedirs(label_save_path)

        valid_datamask_save_path = os.path.join(
            config["datamodule"]["root_dir"], f"{split}_valid_datamasks_poles"
        )
        if not os.path.exists(valid_datamask_save_path):
            os.makedirs(valid_datamask_save_path)

        pool = multiprocessing.Pool(config["dataprep"]["num_workers"])
        pool.starmap(
            create_and_save_mask,
            [
                (
                    img_fn,
                    poly_crs,
                    shapes,
                    types,
                    label_save_path,
                    valid_datamask_save_path,
                )
                for img_fn in img_fns
            ],
        )
        pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file_path",
        type=str,
        default="pole_config.yaml",
        help="Path to config.yaml file",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config_file_path)

    main(config)
