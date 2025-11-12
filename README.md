# SAM Matting Prompt Adapter

A research prototype for foreground matting that adapts the [Segment-Anything Model (SAM)](https://segment-anything.com/) with task-specific prompt adapters. The project focuses on combining natural images and medical imagery, re-using SAM's ViT backbone while adding lightweight adapter layers and a dedicated matting decoder.

## Repository Layout

```
sam_matting/
├── evaluation_results/        # Saved validation metrics, logs, and visualizations
├── matting-tzc/               # Source code for data, model, and training
│   ├── base/                  # Device / experiment helpers
│   ├── config/                # YAML experiment configurations
│   ├── criterion/             # Loss functions for matting
│   ├── data/                  # Data pipelines and dataset utilities
│   └── sam_prompt_adapter/    # Model, training, inference, evaluation scripts
├── pretrained/                # Place SAM checkpoints here (e.g., sam_vit_b_01ec64.pth)
└── test_out/                  # Training checkpoints and inference outputs
```

## Key Components

- **Base utilities** – `matting-tzc/base/base.py` exposes `setup_devices()` for pinning CUDA devices and defines the experiment naming convention used by downstream scripts.【F:matting-tzc/base/base.py†L1-L6】
- **Experiment configuration** – `matting-tzc/config/cod-sam-vit-b.yaml` declares dataset folders, normalization, optimizer, and SAM encoder hyper-parameters. Update the dataset and checkpoint paths before running locally.【F:matting-tzc/config/cod-sam-vit-b.yaml†L1-L86】
- **Data processing** – `matting-tzc/data/data_processor.py` merges natural and medical image batches, converts trimaps to mask prompts, and normalizes prompts for SAM input.【F:matting-tzc/data/data_processor.py†L1-L138】
- **Model definition** – `matting-tzc/sam_prompt_adapter/sam_prompt_adapter.py` wires the ViT image encoder, adapter injection, matting decoder, and multi-loss criterion into the `SAMAdapterPrompt` module.【F:matting-tzc/sam_prompt_adapter/sam_prompt_adapter.py†L1-L142】
- **Training entry point** – `matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py` loads configuration, freezes non-adapter encoder weights, applies grouped learning rates, and logs metrics such as MSE and SAD during validation.【F:matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py†L1-L125】
- **Inference utility** – `matting-tzc/sam_prompt_adapter/inference.py` restores a trained checkpoint, prepares image/mask prompts, and generates alpha mattes with optional visualization.【F:matting-tzc/sam_prompt_adapter/inference.py†L1-L113】

## Environment Setup

1. Create a Python environment (Python ≥ 3.8) with PyTorch and CUDA support.
2. Install the dependencies required by SAM and the matting decoder (PyTorch, torchvision, numpy, OpenCV, tqdm, Pillow, matplotlib, PyYAML). Adapt or create a `requirements.txt` as needed.
3. Download a SAM checkpoint (e.g., `sam_vit_b_01ec64.pth`) and place it under `pretrained/`.
4. Adjust absolute paths hard-coded in scripts (e.g., dataset roots in the YAML config, `sys.path` overrides in `data_processor.py`, and inference defaults) to match your environment.

## Preparing Data

- Configure training/validation/test datasets in `config/cod-sam-vit-b.yaml`. The example assumes paired image & alpha folders for both natural and camouflaged object datasets.【F:matting-tzc/config/cod-sam-vit-b.yaml†L1-L42】
- `DataProcessor` expects trimaps or alpha mattes to create mask prompts; ensure each dataset wrapper returns the necessary keys (`image`/`inp`, `alpha`/`gt`, and optional `mask` or `trimap`).【F:matting-tzc/data/data_processor.py†L17-L105】

## Training

```bash
cd sam_matting/matting-tzc
python sam_prompt_adapter/sam_prompt_train_A.py \
    --config config/cod-sam-vit-b.yaml
```

During startup the script loads `SAMAdapterPrompt`, freezes non-adapter ViT parameters, and creates grouped optimizer settings with higher learning rates for adapter modules.【F:matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py†L64-L116】 Training progress is logged with MSE and SAD metrics per validation epoch.【F:matting-tzc/sam_prompt_adapter/sam_prompt_train_A.py†L28-L61】

## Inference

```bash
cd sam_matting/matting-tzc
python sam_prompt_adapter/inference.py \
    --checkpoint ./test_out/checkpoint_last.pth \
    --input_dir <image_dir> \
    --mask_dir <mask_dir> \
    --output_dir ./test_out/inference_out
```

The script loads the checkpoint, prepares image and binary mask prompts (falling back to full foreground masks when unavailable), and exports predicted alpha mattes. Visualization can be enabled for qualitative inspection.【F:matting-tzc/sam_prompt_adapter/inference.py†L41-L113】

## Evaluation

Use `sam_prompt_adapter/evaluate.py` to compute metrics such as Dice, SAD, MSE, and gradient penalties on validation sets. Customize paths to predictions and ground-truth mattes before running.

## Tips & Further Development

- Replace absolute `sys.path` additions with relative imports or environment variables for improved portability.【F:matting-tzc/data/data_processor.py†L9-L18】【F:matting-tzc/sam_prompt_adapter/inference.py†L20-L29】
- Extend `MattingCriterion` if you need alternative loss combinations or weighting schemes.
- Experiment with adapter depth (`tuning_stage`, `prompt_type`, etc.) in the YAML config to balance accuracy and computational cost.【F:matting-tzc/config/cod-sam-vit-b.yaml†L43-L74】
- Support distributed training by integrating PyTorch DDP/Lightning wrappers around the current scripts.

---

This README summarizes the workflow for preparing data, training SAM Adapter models, and running inference/evaluation for image matting tasks.
