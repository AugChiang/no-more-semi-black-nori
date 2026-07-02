## Training Parameters
- seed: random seed code
- epochs: Number of complete passes through the generated training patches.

### Model
- width: Base feature-channel width of the restoration model.
- middle_blocks: Number of processing blocks in the model bottleneck.
- color_mode: Image channels used for training: gray or rgb.

### Data
- data: Clean image file or directory; subdirectories are scanned recursively.
- batch_size: Patches processed per optimizer update.
- patch_size: Width and height of each square crop in pixels.
- num_patches: Random synthetic patches generated per epoch.
- validation_split: Fraction of clean source pages held out for validation.
- lr: Initial AdamW learning rate.
- workers: DataLoader worker processes; zero loads data in the training process.
- screentone_probability: Probability of adding synthetic clean screentone to a target crop.

### Augmentor
#### screentone: `ScreentoneSynthesizer` constructor arguments.
  - probability: Chance of adding clean synthetic tone to a training target.
  - region_count: Inclusive number of tone regions per augmented crop.
  - strength: Minimum and maximum tone opacity.

#### artifact_augmentor: `BlackArtifactAugmentor` constructor arguments.
  - alpha_range: Minimum and maximum opacity of generated black overlays.

#### Inclusive shape-count ranges per crop:
  - stripe_count: [m, M] 
  - bar_count: [m, M]
  - stain_count: [m, M]
#### Independent probabilities of generating each artifact type.
  - p_stripes: 0.88
  - p_bars: 0.58
  - p_stains: 0.45

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