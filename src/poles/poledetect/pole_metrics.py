# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""""
This script help compute the pole metrics for the pole detection task.
"""
import argparse
import os

import fiona
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.mask
from omegaconf import OmegaConf
from rasterio.features import shapes
from shapely.geometry import Point, Polygon, box


def polygonize(preds_fn, noise_threshold=1):
    """Polygonize the predictions

    Args:
        preds_fn (array): Rasterized predictions
        noise_threshold (int, optional): Threshold to filter out small artifacts. Defaults to 1.

    Returns:
        results (dataframe): Polygonized predictions
    """
    with rasterio.open(preds_fn) as src:
        array = src.read()
        transform = src.transform
        crs = src.crs

    results = (
        {"predictions": v, "geometry": s["coordinates"][0]}
        for i, (s, v) in enumerate(shapes(array, transform=transform))
    )

    results = pd.DataFrame(results)
    results.geometry = results.geometry.apply(Polygon)
    results = gpd.GeoDataFrame(results, geometry=results.geometry)
    results.set_crs(crs, inplace=True)
    results = results[(results.area > noise_threshold) & (results.predictions == 1)]
    results["centroid"] = results.centroid
    return results


def get_valid_labels(image_fn, label_fn, buffer=10):
    """Get labels that are valid for the given image

    Args:
        image_fn (str): Name of the image.
        label_fn (str): Name of the label vector file
        buffer (int, optional): Buffer to apply to the labels. Defaults to 10.
    Returns:
        point_labels (dataframe): Dataframe containing valid labels within the image
    """
    src = rasterio.open(image_fn)
    msk = src.read_masks()
    data_msk = msk[0] & msk[1] & msk[2]

    valid_labels = []
    with fiona.open(label_fn, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        bounds = src.bounds
        geom = box(*bounds)

        for shape in shapes:
            pt = Point(shape["coordinates"])
            if geom.contains(pt):
                lon, lat = shape["coordinates"]
                py, px = src.index(lon, lat)
                if data_msk[py, px] == 255:
                    valid_labels.append(pt)

    valid_labels = gpd.GeoDataFrame(crs=src.crs, geometry=valid_labels)
    valid_labels["buffered_geometry"] = valid_labels.buffer(buffer)
    point_labels = gpd.GeoDataFrame(
        valid_labels, geometry=valid_labels.buffered_geometry
    )
    point_labels.reset_index(inplace=True)
    geoms = point_labels.geometry.unary_union
    point_labels = gpd.GeoDataFrame(geometry=[geoms], crs=valid_labels.crs)
    point_labels = point_labels.explode().reset_index(drop=True)
    return point_labels


def compute_tp_fp_fn(labels, preds):
    """Obtain the true positives, false positives and false negatives

    Args:
        labels (dataframe): Dataframe containing the valid labels points
        preds (dataframe): Dataframe containing the predicted points

    Returns:
        tp, fp,fn (ints): True positives, false positives and false negatives
    """
    import ipdb

    ipdb.set_trace()
    preds.set_crs(labels.crs, inplace=True)
    found_points = gpd.sjoin(preds, labels)

    tp = found_points.shape[0]
    fp = preds.shape[0] - tp
    fn = labels.shape[0] - tp
    return tp, fp, fn


def model_performance(label_fn, preds_fn, image_fn, noise_threshold=1, buffer=10):
    """Compute the model performance for each image

    Args:
        label_fn (str): Points label file name
        preds_fn (str): Prediction raster file name
        image_fn (str): Image file name
        crs (int, optional): _description_. Defaults to 32636.
        buffer (int, optional): _description_. Defaults to 10.
    """
    preds = polygonize(preds_fn, noise_threshold=noise_threshold)
    valid_labels = get_valid_labels(image_fn, label_fn, buffer=buffer)
    tp, fp, fn = compute_tp_fp_fn(valid_labels, preds)

    print(f"File: {image_fn}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Precision: {tp/(tp+fp)}")
    print(f"Recall: {tp/(tp+fn)}")
    print(f"F1: {2*tp/(2*tp+fp+fn)}")
    return tp, fp, fn


def main(config):
    """Compute metrics for the pole detection task for a given tile

    Args:
        config (dict): Dictionary containing the configuration parameters
    """
    model_performance(
        label_fn=config["evaluation"]["label_fn"],
        preds_fn=config["evaluation"]["preds_fn"],
        image_fn=config["evaluation"]["image_fn"],
        noise_threshold=config["evaluation"]["noise_threshold"],
        buffer=config["evaluation"]["buffer"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method_name",
        required=True,
        type=str,
        help="The name of the pole detection experiment.",
    )

    parser.add_argument(
        "--buffer",
        type=int,
        default=10,
        help="Buffer to apply to the labels",
    )
    parser.add_argument(
        "--noise_threshold",
        type=int,
        default=1,
        help="Threshold to filter out small artifacts.",
    )
    parser.add_argument(
        "--label_fn",
        type=str,
        required=True,
        help="Vector file containing the labels",
    )
    parser.add_argument(
        "--image_fn",
        type=str,
        required=True,
        help="Name of corresponding image",
    )
    parser.add_argument(
        "--preds_fn",
        type=str,
        required=True,
        help="Prediction raster file name",
    )

    args = parser.parse_args()
    config = OmegaConf.load(
        os.path.join(
            "checkpoints", args.method_name.lower().replace(" ", "_"), "config.yaml"
        )
    )
    config["evaluation"]["method_name"] = args.method_name
    config["evaluation"]["buffer"] = args.buffer
    config["evaluation"]["noise_threshold"] = args.noise_threshold
    config["evaluation"]["label_fn"] = args.label_fn
    config["evaluation"]["image_fn"] = args.image_fn
    config["evaluation"]["preds_fn"] = args.preds_fn
    main(config)
