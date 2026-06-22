# No More Semi-Black Nori

Restoration code for Japanese monochrome manga pages with screentones. The training pipeline takes clean manga scans, synthesizes semi-transparent black stripes, bars, and stains, and trains a model to recover the original one-channel page without smearing line art or halftone texture.

`data/input_0001.png` is only a bundled smoke-test image. For quality training, use clean monochrome manga pages with real screentones.

## Manga-Focused Defaults

- Images are loaded as grayscale by default (`--color-mode gray`).
- The model checkpoint records `img_channel`, so grayscale and legacy RGB checkpoints both load correctly.
- Training corruption includes thin/large translucent dark overlays, soft stains, and repeated stripe angles.
- Optional clean screentone synthesis adds dot/line tones into light regions during training.
- Losses combine Charbonnier pixel loss, Sobel gradient loss, Laplacian detail loss, weighted frequency loss, and local contrast loss to preserve line art and screentones.

## Install

Use an environment with PyTorch installed, then install the remaining runtime packages:

```bash
pip install opencv-python pillow numpy tqdm torchvision
```

## Prepare Data

Use clean manga images as targets:

```text
clean_manga/
  page_001.png
  page_002.png
  page_003.png
```

Good training data should be uncorrupted or manually cleaned. The scripts create the black artifacts synthetically.

## Train

Default manga-mode training:

```bash
python train.py --data clean_manga --epochs 100 --batch-size 4 --patch-size 256
```

Recommended longer run:

```bash
python train.py \
  --data clean_manga \
  --epochs 300 \
  --num-patches 4000 \
  --batch-size 8 \
  --patch-size 256 \
  --width 32 \
  --middle-blocks 2
```

Small CPU smoke test:

```bash
python train.py --data data/input_0001.png --epochs 1 --num-patches 2 --batch-size 2 --patch-size 64 --width 8 --middle-blocks 1
```

Training writes:

- `checkpoints/best_model.pth`
- `checkpoints/latest_model.pth`
- `samples/epoch_*.png`

Sample grids are arranged as corrupted input, restored output, and clean target.

## Restore

Restore with a trained grayscale manga checkpoint:

```bash
python infer.py \
  --input corrupted_page.png \
  --checkpoint checkpoints/best_model.pth \
  --output restored_page.png \
  --mode model
```

For large manga pages, tiled inference is enabled by default:

```bash
python infer.py --input page.png --output restored.png --tile-size 512 --overlap 64
```

Use `--tile-size 0` to run the full page at once if memory allows.

## Synthetic Demo

To generate a black-artifact input from a clean image and restore it:

```bash
python infer.py \
  --input clean_page.png \
  --checkpoint checkpoints/best_model.pth \
  --output restored_output.png \
  --mode model \
  --add-synthetic \
  --synthetic-output stained_input.png
```

## RGB Compatibility

The repo remains compatible with older RGB checkpoints and experiments:

```bash
python train.py --data clean_images --color-mode rgb
```

For manga, keep the default grayscale mode unless you specifically need color pages.

## Files

- `dataset.py`: manga dataset, screentone target synthesis, and black-artifact generator.
- `model.py`: one-channel-first Dual-Domain NAFNet-style restoration model.
- `train.py`: manga-oriented training loop and texture-preserving losses.
- `infer.py`: grayscale/RGB checkpoint loading, tiled inference, and classical fallback.
