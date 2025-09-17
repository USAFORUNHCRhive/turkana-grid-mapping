# Introduction
** Jump to: [Setup](#setup) | [Directory Structure](#directory-structure) | [Data Acquisition](#data-acquisition) | [Sample Tutorial](#sample-tutorial) | [Acknowledgments](#acknowledgements) | [License](#license) | [Contributing](#contributing)

Mapping electrical infrastructure can support measuring access to electricity and can help identify opportunities to improve or extend exisiting power infrastructure.
High resolution orthorectified imagery (such as drone imagery) can be used to map power distribution infrastructure. The figure below shows a sample power distribution network
that can be mapped using this repository. Here, power infrastructure mapping across the distribution network is done by:
1) Detecting electrical poles
2) Segmenting power distribution lines

![Examples of power infrastructure](src/figures/power_infrastructure_examples.png)

## Setup

Clone this repo and install the conda environment
```
git clone <REPO-URL>
cd <REPO-NAME>
conda env create -f environment.yml
conda activate gridmapper
```
Requirements:
- Linux recommended (environment includes Linux-specific packages).
- NVIDIA GPU with CUDA 11.7 (8GB+ VRAM recommended).
- For CPU-only users or Windows/Mac users, see Tips in the tutorial for adjustments to the environment.

### Directory Structure
```
.
├── environment.yml
├── LICENSE.md
├── README.md
├── TUTORIAL.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
└── src
    ├── figures
    │   └── power_infrastructure_examples.png
    ├── lines
    │   ├── data
    │   │   ├── dataloader_line.py
    │   │   ├── line_dataprep.py
    │   │   ├── line_tile_dataset.py
    │   │   └── streaming_line_dataset.py
    │   ├── line_config.yaml
    │   ├── line_inference.py
    │   ├── lineseg
    │   │   ├── base_network.py
    │   │   ├── line_tasks.py
    │   │   └── unet.py
    │   ├── line_train.py  
    │   └──notebooks
    │       └── 01-00-visualize_patches.ipynb
    └── poles
        ├── data
        │   ├── dataloader_pole.py
        │   ├── pole_dataprep.py
        │   ├── pole_tile_dataset.py
        │   └── streaming_pole_dataset.py
        ├── pole_config.yaml
        ├── poledetect
        │   ├── fcn8_resnet.py
        │   ├── lcfcn_loss.py
        │   ├── pole_metrics.py
        │   ├── pole_tasks.py
        ├── pole_inference.py
        └── pole_train.py
```

## Data Acquisition
This project uses TIFF files of aerial imagery, corresponding point vector labels (for electrical poles) and linestring vector
labels (for electrical lines), as GeoJSON or GeoPackage files. You can request a small sample dataset hosted on Azure blob storage,
collected by USA for UNHCR and the Humanitarian OpenStreetMap Team:

1. Request a data download URL from USA for UNHCR.

2. Once you have the URL, you can use it to access the data on Azure blob storage. Here's an example of how you can download the data with `azcopy`:

```
azcopy copy --recursive <AZURE_BLOB_URL> <DESTINATION_PATH>
```

3. The downloaded raw data folder contains `power_infrastructure_demo` data that can be used to follow the sample [tutorial](TUTORIAL.md).

4. After downloading the data, set the right paths in the config files (src/poles/pole_config.yaml, src/lines/line_config.yaml) as described in the tutorial.

## Sample Tutorial
A sample tutorial walking through the project can be found in [TUTORIAL.md](TUTORIAL.md).

## Acknowledgements
This project was made possible through the collaboration of several organizations: 
[USA for UNHCR](https://www.unrefugees.org/), [UNHCR](https://www.unhcr.org/), [Humanitarian OpenStreetMap Team](https://www.hotosm.org/), and the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/research/group/ai-for-good-research-lab/).

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

If you're interested in contributing to this project check out the [contributing guide](CONTRIBUTING.md) for more details. We also welcome ideas for other ways to contribute - please reach out to hive@unrefugees.org.

