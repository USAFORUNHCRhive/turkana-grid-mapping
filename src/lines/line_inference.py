# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This script is used to run inference on the line segmentation model. Some examples are provided below based on the config.yaml file.
"""

# Import libraries
import argparse
import glob
import os

import numpy as np
import rasterio
import scipy
import torch
import torchvision
from data.line_tile_dataset import TileInferenceDataset
from lineseg.line_tasks import SemanticSegmentationTask
from omegaconf import OmegaConf


def create_downsampled_array(array, kernel_size):
    """Downsample the label mask array to a size array.shape[0]//kernel_size, array.shape[1]//kernel_size

    Args:
        array (array): Input mask array to downsample
        kernel_size (int, optional): The kernel size for downsampling. Defaults to 8.

    Returns:
        downsampled_array (array): Downsampled array given kernel size
    """
    downsampled_array = np.zeros(
        (
            array.shape[0] // kernel_size,
            array.shape[1] // kernel_size,
        )
    )
    for ic in range(0, array.shape[0], kernel_size):
        for jc in range(0, array.shape[1], kernel_size):
            # check that a small block contains a line
            if (
                np.sum(
                    array[
                        ic : ic + kernel_size,
                        jc : jc + kernel_size,
                    ]
                )
                > 0
            ):
                cur_pos_y, cur_pos_x = (
                    ic // kernel_size,
                    jc // kernel_size,
                )
                downsampled_array[cur_pos_y, cur_pos_x] = 1
    return downsampled_array


def get_tp_fp_fn(y_true, y_pred, kernel_size=8):
    """Get true positives, false positives, and false negatives for a given image

    Args:
        y_true (array): Ground truth label mask
        y_pred (array): Predicted label mask
        kernel_size (int, optional): Downsampling factor. Defaults to 8.

    Returns:
        tp, fp, fn (int): True positives, false positives, and false negatives
    """
    gt_array = create_downsampled_array(y_true, kernel_size)

    assert gt_array.shape == y_pred.shape
    tp = np.sum((gt_array == 1) & (y_pred == 1))
    fp = np.sum((gt_array == 0) & (y_pred == 1))
    fn = np.sum((gt_array == 1) & (y_pred == 0))
    return tp, fp, fn


def main(config):
    """
    Main inference routine specific for line segmentation using the config file."""
    NUM_WORKERS = config["inference"]["num_workers"]
    CHIP_SIZE = config["inference"]["chip_size"]
    CHANNELS = config["learning"]["in_channels"]
    PADDING = config["inference"]["padding"]
    assert PADDING % 2 == 0
    HALF_PADDING = PADDING // 2
    CHIP_STRIDE = CHIP_SIZE - PADDING
    EXPERIMENT_NAME = config["evaluation"]["method_name"].lower().replace(" ", "_")
    OUTPUT_DIR = config["inference"]["output_dir"]
    BATCH_SIZE = config["inference"]["batch_size"]
    KERNEL_SIZE = config["datamodule"]["segm_filter_size"]

    compute_metrics = (
        True if config["inference"]["label_mask_dir"] is not None else False
    )
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    device = torch.device(
        "cuda:%d" % (config["inference"]["devices"])
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SemanticSegmentationTask(**config.learning)

    ckpt_dir_path = f"checkpoints/{EXPERIMENT_NAME}"
    checkpoint = torch.load(os.path.join(ckpt_dir_path, "best_model-v2.ckpt"))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model = model.to(device)

    filenames = [k for k in glob.glob(config["inference"]["test_image_dir"] + "*.tif")]
    for input_fn in filenames:
        #
        print(f"Starting inference for image: {input_fn}")
        if not os.path.exists(os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)):
            os.makedirs(os.path.join(OUTPUT_DIR, EXPERIMENT_NAME))
        output_fn = os.path.join(
            OUTPUT_DIR, EXPERIMENT_NAME, str(os.path.basename(input_fn))
        )
        if os.path.exists(output_fn):
            continue

        with rasterio.open(input_fn) as f:
            input_width, input_height = f.width, f.height
            input_profile = f.profile.copy()

        transform_set = torchvision.transforms.Compose(
            [
                lambda x: np.rollaxis(x.astype(np.float32), 2, 0),
                lambda x: x / 255.0,
                lambda x: torch.from_numpy(x),
            ]
        )

        dataset = TileInferenceDataset(
            input_fn,
            chip_size=CHIP_SIZE,
            stride=CHIP_STRIDE,
            channels=CHANNELS,
            transform=transform_set,
            verbose=True,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # -------------------
        # Run inference
        # -------------------
        output = np.zeros((input_height, input_width), dtype=np.uint8)
        for i, (data, coords) in enumerate(dataloader):
            data = data.to(device)
            tmp_output = np.zeros(
                (data.shape[0], CHIP_SIZE, CHIP_SIZE), dtype=np.float32
            )
            with torch.no_grad():
                t_output = model(data).argmax(axis=1).cpu().numpy()
                for batch_idx in range(t_output.shape[0]):
                    tmp_output[batch_idx] = scipy.ndimage.zoom(
                        t_output[batch_idx], KERNEL_SIZE, order=1
                    )

            for j in range(tmp_output.shape[0]):
                y, x = coords[j]
                output[
                    y + HALF_PADDING : y + CHIP_SIZE - HALF_PADDING,
                    x + HALF_PADDING : x + CHIP_SIZE - HALF_PADDING,
                ] = tmp_output[
                    j, HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING
                ]

        print("Saving output to: %s" % output_fn)
        output_profile = input_profile.copy()
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(output, 1)

        if compute_metrics:
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * (precision * recall) / (precision + recall)

            print(f"Metrics for image : {input_fn}")
            print(f"Precision : {precision}")
            print(f"Recall : {recall}")
            print(f"F1 score : {f1_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method_name",
        required=True,
        type=str,
        help="The name of the model experiment.",
    )
    parser.add_argument("--gpu", type=int, required=True)

    args = parser.parse_args()
    config = OmegaConf.load(
        os.path.join(
            "checkpoints", args.method_name.lower().replace(" ", "_"), "config.yaml"
        )
    )
    config["evaluation"]["method_name"] = args.method_name
    config["inference"]["devices"] = args.gpu if args.gpu != -1 else -1
    config["inference"]["label_mask_dir"] = args.label_mask_dir
    main(config)
