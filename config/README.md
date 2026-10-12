## Training Parameters
- seed: random seed code
- epochs: Number of complete passes through the generated training patches.

### Model
- width: Base feature-channel width of the restoration model.
- middle_blocks: Number of processing blocks in the model bottleneck.
- encoder_blocks, decoder_blocks: NAF block counts per stage; the two lists must have equal length.
- use_fft: Enable the older FFT bottleneck for compatible checkpoints; false uses the NAFNet backbone alone.
- amp: Use CUDA mixed precision. BF16 is preferred when the GPU supports it; CPU training stays FP32.
- color_mode: Image channels used for training: gray or rgb.
- nafnet_pretrained_weights: Official NAFNet checkpoint used to initialize a new run. Defaults to `pretrain-weights/NAFNet-SIDD-width32.pth`; null or `--no-pretrained` trains from scratch. The default stage layout matches this checkpoint. RGB boundary layers are converted when `color_mode` is gray.
- `--resume`: Restores this repo's full training checkpoint instead of loading the pre-trained initialization.

### Data
- data: Clean image file or directory; subdirectories are scanned recursively.
- batch_size: Patches processed per optimizer update.
- patch_size: Width and height of each square crop in pixels.
- num_patches: Random synthetic patches generated per epoch.
- validation_split: Fraction of NudeNet-detected pages held out for validation.
- Validation corruptions are fixed per image and streamed each epoch.
- lr: Initial AdamW learning rate.
- workers: DataLoader worker processes; zero loads data in the training process.
- screentone.probability: Probability of adding synthetic clean screentone to a target crop.

### NudeNet region guidance
- nudenet_weights: Required path to NudeNet 640m `.pt` weights; default `NudeNet/chpt/640m.pt`.
- nudenet_cache: JSON path for cached detections on clean source images.
- nudenet_confidence: Minimum score for the two exposed genitalia labels.
- nudenet_resolution: Detector input size; 640 matches the 640m checkpoint.
- nudenet_batch_size: Clean images processed in each detection batch.
- region_crop_probability: Chance to center a crop on its selected region; default 1.0. A random crop without a visible box produces no mask.
- Only images with a NudeNet detection enter the training and validation split. An empty detection set stops training.

### Augmentor
#### screentone: `ScreentoneSynthesizer` constructor arguments.
  - probability: Chance of adding clean synthetic tone to a training target.
  - region_count: Inclusive number of tone regions per augmented crop.
  - strength: Minimum and maximum tone opacity.

#### artifact_augmentor: `BlackArtifactAugmentor` constructor arguments.
  - alpha_range: Minimum and maximum opacity of generated black overlays.
  - bar_count: Inclusive number of separate bar groups per visible detected box; default is one.
  - bars_per_group: Inclusive number of parallel rectangles in each group.
  - bar_thickness: Minimum and maximum bar thickness as fractions of the shorter crop side; shrinks when necessary to fit a box.
  - bar_gap: Minimum and maximum clear gap as multiples of bar thickness; each adjacent pair gets its own sampled gap.
  - guided_bar_width: Minimum and maximum horizontal bar width as fractions of the visible NudeNet box width; defaults to 0.70 to 0.90.
  - guided_angle_range: Minimum and maximum group angle in degrees from horizontal; defaults to -8 to +8 and must stay within (-45, 45).
  - guided_center_jitter: Maximum horizontal and vertical shift of the group center as a fraction of box width and height; default 0.10.
  - guided_bar_shift: Maximum horizontal shift of each bar relative to the group as a fraction of box width; default 0.03.

Bar groups use one angle and opacity per group. Every rectangle stays inside
its detected box, and groups are rejected if they touch existing bars. When one
box contains another, the wider enclosing box is used. A box too small to fit
the sampled bars may get fewer bars or no mask.

### Losses
- artifact_weight: Extra reconstruction weight within synthetic artifact masks.
- freq_weight: Frequency-domain detail preservation weight.
- gradient_weight: Edge-gradient preservation weight.
- laplacian_weight: Fine-detail Laplacian preservation weight.
- contrast_weight: Local screentone and line-contrast preservation weight.

### Output
- checkpoint_dir: Directory for latest and best model checkpoints.
- sample_dir: Directory for corrupted/output/target preview grids.
- sample_every: Epoch interval between saved preview grids.

### Early Stopper
- early_stop_patience: Unimproved preview epochs allowed; zero disables early stopping.
- early_stop_min_delta: Minimum preview-loss decrease counted as improvement.
