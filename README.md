# No More Semi-Black Nori

Train an image restoration model to remove semi-transparent black censoring bars
from monochrome manga. Training needs **clean pages**: the dataset creates the
corrupted input and keeps the clean page as its target.

## Pipeline

1. `dataset.py` loads clean pages, crops training patches, and optionally adds
   clean screentones. It draws translucent bars with varied gaps and angles.
2. `nudenet_regions.py` caches detections for `FEMALE_GENITALIA_EXPOSED` and
   `MALE_GENITALIA_EXPOSED`. Training uses pages with detected regions; bars stay
   inside each visible box, and nested boxes use the wider enclosing box.
   Crops without a visible box receive no bar mask.
3. `model.py` restores the corrupted patch with a configurable NAFNet encoder,
   bottleneck, and decoder. By default, training initializes it from the local
   SIDD-width32 checkpoint. RGB input/output weights are converted for grayscale
   pages; the older FFT bottleneck remains optional for legacy models.
4. `train.py` minimizes a pixel loss plus artifact, edge, frequency, and local
   contrast terms. Validation uses repeatable synthetic masks and streams pages
   each epoch. On CUDA, training uses BF16 mixed precision when supported.
5. `infer.py` loads the saved architecture and restores a full page or blends
   overlapping tiles. It also provides a classical fallback and a synthetic
   corruption demo.

The NAF block design follows the [official NAFNet implementation](https://github.com/megvii-research/NAFNet/tree/main/basicsr/models/archs).
The pre-trained weights initialize a new manga-restoration run; training then
updates them on the synthetic bar-removal task.

## Setup

Use Python 3.9+ with PyTorch installed. In the provided Conda environment:

```bash
conda activate ml
pip install opencv-python pillow numpy pyyaml tqdm torchvision
```

For the default NudeNet guidance, install `ultralytics`. The default config
expects `pretrain-weights/NudeNet-640m.pt` and `pretrain-weights/NAFNet-SIDD-width32.pth`
relative to the repository root. Place the downloaded NudeNet `640m.pt` in that
location; the NudeNet source clone is not required for training.

## Train

Put clean `.png`, `.jpg`, or other supported images in a directory. Subfolders
are scanned recursively. Training selects CUDA automatically when available and
requires at least one NudeNet detection:

```bash
python train.py --data clean_manga --epochs 100 --batch-size 4 --patch-size 256
```

Both checkpoints load by default. To choose different weights:

```bash
python train.py --data clean_manga \
  --nafnet-pretrained-weights pretrain-weights/NAFNet-SIDD-width32.pth \
  --nudenet-weights pretrain-weights/NudeNet-640m.pt
```

For a minimal run on a CUDA machine, replace `clean_manga/positive_page.png`
with one clean image that NudeNet detects:

```bash
python train.py --data clean_manga/positive_page.png --epochs 1 --num-patches 2 \
  --batch-size 2 --patch-size 64 --width 8 --middle-blocks 1 \
  --encoder-blocks 1 1 --decoder-blocks 1 1 --no-pretrained
```

Settings live in [`config/default.yaml`](config/default.yaml). `--config` loads
another YAML file; scalar CLI flags override its values. Key settings:

| Setting | Purpose |
| --- | --- |
| `width`, `middle_blocks`, `encoder_blocks`, `decoder_blocks` | Set NAFNet capacity and stage layout. Encoder and decoder lists must have equal length. |
| `nafnet_pretrained_weights` / `--no-pretrained` | Initialize a new run from the official SIDD-width32 checkpoint, or train from random weights. The model layout must match the checkpoint. |
| `nudenet_weights` | Detector weights required for guided synthetic masks. |
| `use_fft` / `--use-fft` | Enable the legacy FFT bottleneck; new training defaults to `false`. |
| `amp` / `--no-amp` | Toggle CUDA mixed precision; CPU uses FP32. |
| `artifact_augmentor.guided_bar_width` | Guided bar width as a fraction of the visible NudeNet box width; default `[0.70, 0.90]`. |
| `artifact_augmentor.guided_angle_range` | Random angle in degrees for each bar group; default `[-8, 8]`. |
| `region_crop_probability` | Chance of centering a crop on its selected detection; default `1.0`. |
| `validation_split`, `sample_every` | Set held-out pages and preview frequency. |

See [`config/README.md`](config/README.md) for all configuration fields.
Training writes `latest_model.pth`, `best_model.pth`, and loss histories to
`checkpoints/`; preview grids go to `samples/`. Resume with
`--resume checkpoints/latest_model.pth`. Checkpoints record the model layout,
optimizer, scheduler, and mixed-precision state. `--resume` takes precedence over
the NAFNet pre-trained path and continues the saved optimizer state.

To inspect the separate NTD11 YOLO segmentation checkpoint, run
`python preview_ntd11.py path/to/page.png`. It saves a comparison against cached
NudeNet boxes, raw NTD11 candidates, a genital-only mask, and detection scores
under `samples/ntd11_inference/`. The accepted genital threshold comes from
`mosaic-tool/config.yaml`; this preview does not change the training detector.

**Detector decision (2026-10-12): Skip NTD11 for monochrome manga.** In the
11-page spot check, it accepted genital regions on only 2 of 8 pages with
cached NudeNet boxes. It returned no candidates on all 4 monochrome pages in
that group at 640 px and raw confidence 0.01. Raising the size to 1024 px and
lowering raw confidence to 0.001 produced only weak candidates, with none
meeting mosaic-tool's 0.50 genital threshold. Keep NudeNet as the default
region guide for this dataset. Review `samples/ntd11_inference/overview.png`
and `samples/ntd11_inference_lowconf_1024/overview.png` for the visual checks;
this small comparison is not a ground-truth benchmark.
![link](https://note.com/major_myrtle6198/n/nf192f114cf1d)

## Restore

```bash
python infer.py --input corrupted_page.png --checkpoint checkpoints/best_model.pth \
  --output restored.png --mode model
```

Use `--tile-size 512 --overlap 64` for large pages, or `--tile-size 0` for a
single full-page pass. `--mode classical` needs no checkpoint. For a synthetic
input preview, add `--add-synthetic --synthetic-output stained_input.png`
and `--nudenet-weights pretrain-weights/NudeNet-640m.pt`. The input must have a detection.

The model supports grayscale (`--color-mode gray`, default) and RGB training.
Inference reads the channel count and architecture from the checkpoint.
