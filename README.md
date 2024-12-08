# Image-to-Image Translation Tools

This repository contains supplementary tools and scripts used for pre-processing and post-processing in image-to-image translation projects. These tools were used alongside Pix2Pix and CycleGAN pipelines as part of a thesis project. Below is an overview of the provided scripts, their functionalities, and how to use them.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Scripts Overview](#scripts-overview)
  - [1. Holistic Nested Edge Detection (`holisticNestedEdgeDetection.py`)](#1-holistic-nested-edge-detection)
  - [2. Image Segmentation (`image_segmentation.py`)](#2-image-segmentation)
  - [3. Background Removal (`remove_background.py`)](#3-background-removal)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

---

## Overview

This repository contains three primary scripts:

1. **Holistic Nested Edge Detection**: Generates edge-detected versions of input images using a Holistically-Nested Edge Detection (HED) model.
2. **Image Segmentation**: Performs semantic segmentation using the SAM2 model and ResNet50-based classification.
3. **Background Removal**: Removes the background from images and replaces it with a white background using the `rembg` library.

---

## Requirements

Ensure the following software and libraries are installed:

- Python 3.8 or above
- Required Python libraries (install via `pip`):
  - `opencv-python`
  - `numpy`
  - `torch`
  - `torchvision`
  - `Pillow`
  - `rembg`
  - `matplotlib`
- Pre-trained model files:
  - `deploy.prototxt` and `hed_pretrained_bsds.caffemodel` for HED
  - Checkpoints and configurations for SAM2

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/developsomethingcool/additional_scripts
   cd image-to-image-tools

2. Install the required Python libraries
3. Place the required model files in the appropriate directories:

- **HED models**: Place `deploy.prototxt` and `hed_pretrained_bsds.caffemodel` in the root directory.
  - Download the models from the [Holistically-Nested Edge Detection (HED)](https://github.com/s9xie/hed) GitHub repository.

- **SAM2 models**: Place `sam2.1_hiera_large.pt` and its configuration file (`sam2.1_hiera_l.yaml`) in the specified directories.
  - Learn more and download from the [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything).

- **Rembg**: Install the library from its [Rembg GitHub repository](https://github.com/danielgatis/rembg).

## Scripts Overview

1. Holistic Nested Edge Detection (holisticNestedEdgeDetection.py)

Functionality:

    Detects edges in input images using a pre-trained Holistically-Nested Edge Detection (HED) model.
    Outputs resized edge-detected images to a specified directory.

Usage:

    Update dataset_dir and updated_dataset_dir to specify the input and output directories.
    Ensure deploy.prototxt and hed_pretrained_bsds.caffemodel are present.

2. Image Segmentation (image_segmentation.py)

Functionality:

    Performs semantic segmentation using the SAM2 model.
    Classifies each segmented region using a ResNet50-based classifier.
    Generates semantic maps for visualization and saves the results.

Usage:

    Update dataset_dir and results_dir to specify the input and output directories.
    Place SAM2 model checkpoint and configuration file in the appropriate paths.

3. Background Removal (remove_background.py)

Functionality:

    Removes the background of images and replaces it with a white background.
    Uses the rembg library for background removal.

Usage:

    Update dataset_dir and updated_dataset_dir to specify the input and output directories.
    Ensure the rembg library is installed.

## Usage

1. Place your input images in the respective `dataset_dir` for each script.

2. Run the script:
   ```bash
   python <script_name>.py


## Acknowledgments

- [Holistically-Nested Edge Detection (HED)](https://github.com/s9xie/hed)
- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [Rembg](https://github.com/danielgatis/rembg)

For any issues, suggestions, or contributions, feel free to open an issue or a pull request.

