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
from sklearn.neighbors import BallTree
from fiona import transform


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


def get_valid_labels(image_fn, label_fn):
    """Get labels that are valid for the given image

    Args:
        image_fn (str): Name of the image.
        label_fn (str): Name of the label vector file
    Returns:
        point_labels (dataframe): Dataframe containing valid labels within the image
    """
    src = rasterio.open(image_fn)
    msk = src.read()  # img_fp.read_masks()
    src_crs = src.crs
    data_msk = ((msk[0] > 0) & (msk[1] > 0) & (msk[2] > 0)).astype(int)

    valid_labels = []
    with fiona.open(label_fn, "r") as shapefile:
        dst_crs = shapefile.crs
        shapes = [feature["geometry"] for feature in shapefile]
        bounds = src.bounds
        geom = box(*bounds)

        for shape in shapes:
            pt = Point(shape["coordinates"])

            point = transform.transform(
                dst_crs, src_crs, [shape["coordinates"][0]], [shape["coordinates"][1]]
            )
            point = Point(point[0][0], point[1][0])

            if geom.contains(point):
                lon, lat = point.xy
                py, px = src.index(lon, lat)
                if data_msk[py, px] == 1:
                    valid_labels.append(pt)
    valid_labels = gpd.GeoDataFrame(crs=dst_crs, geometry=valid_labels)
    return valid_labels


def get_nearest(src_points, candidates, k_neighbors=1):
    """
    Source: https://autogis-site.readthedocs.io/en/2019/notebooks/L3/nearest-neighbor-faster.html
    Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric="euclidean")

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def compute_tp_fp_fn(labels, preds, buffer=10):
    """Obtain the true positives, false positives and false negatives

    Args:
        labels (dataframe): Dataframe containing the valid labels points
        preds (dataframe): Dataframe containing the predicted points
        buffer (int): Buffer to apply to the labels

    Returns:
        tp, fp,fn (ints): True positives, false positives and false negatives
    """
    labels.to_crs(preds.crs, inplace=True)
    preds.reset_index(inplace=True)
    labels.reset_index(inplace=True)

    pred_data = preds.centroid.apply(lambda row: (row.xy[0][0], row.xy[1][0])).to_list()
    true_data = labels.geometry.apply(
        lambda row: (row.xy[0][0], row.xy[1][0])
    ).to_list()

    closest, dist = get_nearest(src_points=pred_data, candidates=true_data)
    closest_points = labels.loc[closest]

    closest_points["dist"] = dist
    closest_points["label_index"] = closest_points.index.tolist()
    closest_points["gt_labels"] = closest
    closest_points = closest_points.reset_index(drop=True)

    # sort by distance and drop duplicates
    precision_tp_strict = closest_points.sort_values(
        by="dist", ascending=True
    ).drop_duplicates("gt_labels")
    precision_tp_strict = precision_tp_strict[precision_tp_strict.dist < buffer].shape[
        0
    ]
    fp_strict = closest_points.shape[0] - precision_tp_strict

    recall_tp = precision_tp_strict
    fn = labels.shape[0] - recall_tp

    return precision_tp_strict, recall_tp, fp_strict, fn


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
    valid_labels = get_valid_labels(image_fn, label_fn)
    precision_tp, recall_tp, fp, fn = compute_tp_fp_fn(valid_labels, preds, buffer)
    precision = precision_tp / (precision_tp + fp)
    recall = recall_tp / (recall_tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    print(f"File: {image_fn}")
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("F1: {:.2f}".format(f1))
    return precision, recall, f1


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
