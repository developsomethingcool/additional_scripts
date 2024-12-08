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
