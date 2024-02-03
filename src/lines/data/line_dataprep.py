# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""""
This script prepares the data for the line segmentation task."""
import argparse
import multiprocessing
import os

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.mask
from omegaconf import OmegaConf
from shapely.geometry import LineString, box
from skimage.draw import line


def create_and_save_mask(img_fn, shapes, types, line_crs, label_save_path):
    """Create and save the masks for the given image.

    Args:
        img_fn (str): The filename of the image to create the masks for.
        shapes (list): A list of shapes to create masks for.
        types (list): A list of types to create masks for.
        line_crs (dict): The crs of the line shapes.
        label_save_path (str): Path to save the masks to.
    """
    print(f"Working on image: {img_fn}")
    img_fp = rasterio.open(os.path.join(img_fn), "r")
    img_crs = img_fp.crs
    geom = box(*img_fp.bounds)
    matches_shapes = []
    matches_types = []
    # Find line strings that are in the image
    for idx, shape in enumerate(shapes):
        warped_shape = [
            fiona.transform.transform(line_crs, img_crs, [coord[0]], [coord[1]])
            for coord in shape["coordinates"]
        ]
        cleaned_warped_shape = [[k[0][0], k[1][0]] for k in warped_shape]
        grid = LineString(cleaned_warped_shape)
        if geom.contains(grid):
            matches_shapes.append(cleaned_warped_shape)
            matches_types.append(types[idx])
            label_data = np.zeros((img_fp.read().shape[1], img_fp.read().shape[2]))
            for i, cur_line in enumerate(matches_shapes):
                for j in range(len(cur_line) - 1):
                    start_lon, start_lat = cur_line[j]
                    stop_lon, stop_lat = cur_line[j + 1]
                    start_py, start_px = img_fp.index(start_lon, start_lat)
                    stop_py, stop_px = img_fp.index(stop_lon, stop_lat)
                    rr, cc = line(start_py, start_px, stop_py, stop_px)
                    if matches_types[i] == "line":
                        label_data[rr, cc] = 1

    output_profile = img_fp.profile.copy()
    output_profile["dtype"] = "uint8"
    output_profile["count"] = 1

    # Save the generated mask
    token = img_fn.split("/")[-1]
    print(f"Saving : {token}")
    with rasterio.open(
        os.path.join(label_save_path, token), "w", **output_profile
    ) as f:
        f.write(label_data, 1)


def main(config):
    """This script prepares the data for the line segmentation task."""

    # Get the line features from the geojson file
    with fiona.open(config["dataprep"]["line_vector_file"], "r") as shapefile:
        line_crs = shapefile.crs
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
            config["datamodule"]["root_dir"], f"{split}_masks_lines"
        )
        if not os.path.exists(label_save_path):
            os.makedirs(label_save_path)

        pool = multiprocessing.Pool(config["dataprep"]["num_workers"])
        pool.starmap(
            create_and_save_mask,
            [
                (
                    img_fn,
                    shapes,
                    types,
                    line_crs,
                    label_save_path,
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
        default="line_config.yaml",
        help="Path to config.yaml file",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config_file_path)

    main(config)
