# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

LaMa (Large Mask Inpainting) is a deep learning model for image inpainting based on Fast Fourier Convolutions (FFC). The key innovation is splitting convolutional channels into local (spatial) and global (Fourier domain) branches, enabling the model to generalize to resolutions far beyond training size (trained at 256x256, generalizes to ~2K). Paper: "Resolution-robust Large Mask Inpainting with Fourier Convolutions" (arXiv:2109.07161).

## Common commands

### Inference

```bash
# Standard prediction via Hydra CLI
python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

# With refinement (higher quality, slower)
python3 bin/predict.py refine=True model.path=$(pwd)/big-lama indir=$(pwd)/LaMa_test_images outdir=$(pwd)/output

# Custom inference script (bypasses Hydra CLI — required for non-ASCII/Chinese paths)
python run_lama.py --images_dir <path> --masks_dir <path> --output_dir <path>
```

### Training

```bash
# Set required env vars first
export TORCH_HOME=$(pwd) && export PYTHONPATH=$(pwd)

# Train LaMa-Fourier on Places
python3 bin/train.py -cn lama-fourier location=places_standard

# Train Big-LaMa
python3 bin/train.py -cn big-lama location=places_standard

# Override config parameters via CLI
python3 bin/train.py -cn lama-fourier data.batch_size=10 run_title=my-title
```

### Evaluation

```bash
# Compute metrics (SSIM, LPIPS, FID) on predictions
python3 bin/evaluate_predicts.py \
    $(pwd)/configs/eval2_gpu.yaml \
    $(pwd)/path/to/eval_dataset/ \
    $(pwd)/path/to/predictions/ \
    $(pwd)/path/to/output_metrics.csv
```

### Generate mask datasets

```bash
python3 bin/gen_mask_dataset.py \
    $(pwd)/configs/data_gen/random_medium_512.yaml \
    /path/to/source_images/ \
    /path/to/output_dataset/ \
    --ext jpg
```

### Setup

```bash
pip install -r requirements.txt
# Model download: https://drive.google.com/drive/folders/1B2x7eQDgecTL0oh3LSIBDGj0fTxs6Ips
# Unzip big-lama.zip into ./big-lama/
```

## Architecture

### Core library: `saicinpainting/`

- **`training/modules/ffc.py`** — The Fast Fourier Convolution implementation. `FFC` is the building block: splits input channels into local (ratio_gin) and global portions. Local paths use regular Conv2d; global paths use `SpectralTransform` (real FFT → conv in frequency domain → inverse FFT). `FFCResNetGenerator` builds the full generator as an encoder-bottleneck-decoder with FFC blocks. `FFCNLayerDiscriminator` is the discriminator variant.

- **`training/trainers/base.py`** — `BaseInpaintingTrainingModule` (PyTorch Lightning). Holds generator, discriminator, losses, evaluator, visualizer. Uses EMA on generator weights (`average_generator`).

- **`training/trainers/default.py`** — `DefaultInpaintingTrainingModule` extends base. The forward pass: mask the image, optionally add noise, concatenate mask channel, feed to generator, blend predicted and known regions with `inpainted = mask * predicted + (1-mask) * image`. Supports rescaling scheduler, constant-area cropping, fake-fakes augmentation.

- **`training/data/datasets.py`** — Training datasets (`InpaintingTrainDataset`, `InpaintingTrainWebDataset`) and validation datasets. Mask generators produce random masks on-the-fly. Data augmentation via Albumentations.

- **`training/losses/`** — L1 loss, adversarial loss (R1/Non-saturating), perceptual loss (VGG-based), feature matching loss, ResNet perceptual loss, segmentation loss.

- **`evaluation/`** — `InpaintingEvaluator` computes SSIM, LPIPS, FID. Supports segmentation-aware variants. `PrecomputedInpaintingResultsDataset` loads pre-generated predictions for offline evaluation.

### Entry points: `bin/`

- **`predict.py`** — Hydra-based inference. Loads model from checkpoint, runs on image-mask pairs from a directory. Masks must be named `image_maskXXX.png` alongside `image.png`. Outputs inpainted images.

- **`train.py`** — Hydra-based training with PyTorch Lightning. Uses DDP for multi-GPU. Saves checkpoints and TensorBoard logs.

- **`evaluate_predicts.py`** — Compares prediction directory against ground truth dataset, outputs CSV metrics.

- **`gen_mask_dataset.py`** — Generates fixed random masks (thick/medium/thin variants) for evaluation datasets.

### Configuration system

Hydra/OmegaConf with YAML configs under `configs/`. Key locations:

- `configs/training/` — Model definitions (`lama-fourier.yaml`, `big-lama.yaml`, etc.) with `defaults:` composing sub-configs for location, data, generator, discriminator, optimizers, trainer.
- `configs/prediction/default.yaml` — Inference settings (dataset format, device, refinement options).
- `configs/data_gen/` — Mask generation presets (`random_thick_512.yaml`, `random_thin_256.yaml`, etc.).
- `configs/training/location/` — Dataset paths per machine (overridden via `location=<name>`).
- Parameters are overridden at CLI: `data.batch_size=10`, `losses.adversarial.weight=5`, etc.

### Model variants (config names)

- `big-lama` — Largest model (18 FFC blocks, ngf=64), Places Challenge
- `big-lama-regular` — Same but standard convolutions instead of FFC (ablation)
- `lama-fourier` — Default LaMa with FFC (9 blocks)
- `lama-regular` — Default LaMa without FFC (ablation)
- `lama_small_train_masks` — Trained with smaller mask area distribution

### Data conventions

- Images and masks in the same directory.
- Mask naming: `{image_name}_mask{XXX}.{ext}`, e.g. `photo1_mask001.png` alongside `photo1.png`.
- Masks are grayscale; white (255) = region to inpaint.
- Images are read as BGR by OpenCV, converted to RGB internally, model outputs RGB.

### `run_lama.py` (local addition)

Custom inference script that bypasses Hydra CLI entirely. Created because Hydra has issues with Chinese character paths on Windows. Directly imports `saicinpainting` modules and replicates the prediction pipeline. Takes separate `--images_dir` and `--masks_dir` arguments and formats them into LaMa's expected directory structure in a temp folder.
