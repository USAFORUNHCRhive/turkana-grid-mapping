# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This script is used to run inference on the pole detection model. Some examples are provided below based on the config.yaml file.
$ python pole_inference.py --method_name 'baseline' --gpu 2
"""

# Import libraries
import argparse
import glob
import os

import numpy as np
import rasterio
import torch
import torchvision
from data.pole_tile_dataset import TileInferenceDataset
from omegaconf import OmegaConf
from poledetect.pole_tasks import SemanticSegmentationTask


def main(config):
    """
    Main inference routine specific for pole detection using the config file."""
    NUM_WORKERS = config["inference"]["num_workers"]
    CHIP_SIZE = config["inference"]["chip_size"]
    CHANNELS = config["learning"]["in_channel"]
    PADDING = config["inference"]["padding"]
    assert PADDING % 2 == 0
    HALF_PADDING = PADDING // 2
    CHIP_STRIDE = CHIP_SIZE - PADDING
    EXPERIMENT_NAME = config["evaluation"]["method_name"].lower().replace(" ", "_")
    OUTPUT_DIR = config["inference"]["output_dir"]
    BATCH_SIZE = config["inference"]["batch_size"]

    device = torch.device(
        "cuda:%d" % (config["inference"]["devices"])
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SemanticSegmentationTask(**config.learning)

    ckpt_dir_path = f"checkpoints/{EXPERIMENT_NAME}"
    checkpoint = torch.load(os.path.join(ckpt_dir_path, "best_model.ckpt"))
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
            with torch.no_grad():
                t_output = model(data).sigmoid()
                t_output = (t_output[:, 0, :, :] > 0.5).cpu().numpy()

            for j in range(t_output.shape[0]):
                y, x = coords[j]
                output[
                    y + HALF_PADDING : y + CHIP_SIZE - HALF_PADDING,
                    x + HALF_PADDING : x + CHIP_SIZE - HALF_PADDING,
                ] = t_output[j, HALF_PADDING:-HALF_PADDING, HALF_PADDING:-HALF_PADDING]

        print("Saving output to: %s" % output_fn)
        output_profile = input_profile.copy()
        output_profile["dtype"] = "uint8"
        output_profile["count"] = 1
        output_profile["nodata"] = 0

        with rasterio.open(output_fn, "w", **output_profile) as f:
            f.write(output, 1)


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
    main(config)
