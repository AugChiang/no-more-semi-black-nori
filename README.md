# Stripe Stripper: Manga De-striping with Dual-Domain NAFNet

A deep learning project to remove semi-transparent black stripes from monochrome manga images while preserving delicate screentone patterns.

## Overview

This project uses a **Dual-Domain NAFNet** (Nonlinear Activation Free Network) architecture. It is specifically optimized for manga restoration by combining:
- **Spatial Branch**: Preserves sharp line art and structures.
- **Frequency Branch (FFT)**: Surgically suppresses periodic noise (stripes) in the Fourier domain without blurring screentones.
- **Focal Frequency Loss**: Ensures high-fidelity reconstruction of periodic textures.

## Features
- **Procedural Augmentation**: Randomly generates non-overlapping black stripes with varying alpha (0.1 - 0.8), widths, and angles.
- **Manga-Optimized**: Specifically designed to handle the high-frequency nature of manga screentones.
- **Efficient**: Purely convolutional architecture, significantly faster than Transformer-based models.

## Installation

Ensure you have a Python environment with PyTorch and CUDA support.

```bash
conda activate ml
pip install opencv-python scikit-image focal-frequency-loss tqdm torchvision
```

## Usage

### 1. Prepare Data
Place your ground truth image (e.g., `sample.webp`) in the root directory.

### 2. Training
To train the model on your sample image:
```bash
python train.py
```
- Check the `samples/` directory to see visual progress during training.
- The best model will be saved in `checkpoints/best_model.pth`.

### 3. Inference
To restore an image using the trained model:
```bash
python infer.py
```
- This will create `stained_input.png` (for demonstration) and `restored_output.png` (the restored result).

## File Structure
- `model.py`: Dual-Domain NAFNet architecture definition.
- `dataset.py`: Synthetic data generator with non-overlapping stripe augmentation.
- `train.py`: Training loop with spatial and frequency losses.
- `infer.py`: Inference script for full-image restoration.
- `sample.webp`: Your target monochrome manga image.

## Architecture Detail
The model integrates **FFT-based Spectral Blocks** in the bottleneck of a NAFNet U-Net. This allow the network to "zero out" the specific frequencies corresponding to the stripes while protecting the broader frequency spectrum of the manga art.
