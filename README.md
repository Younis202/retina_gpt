# 🔬 Retina-GPT: Foundation Model for Retinal Image Analysis

<div align="center">

```
Retina Image → Preprocessing → Vision Encoder → Universal Embedding
       → Multi-Task Heads → Medical Language Model → Clinical Report
```

**Production-Grade AI Pipeline for Clinical Retinal Analysis**

</div>

---

## 🏗️ Architecture Overview

Retina-GPT is a multimodal foundation model that combines:
- **Retina Vision Transformer (RVT)** — specialized ViT backbone for fundus images
- **Multi-Task Heads** — simultaneous classification, segmentation & detection
- **Medical Language Model** — vision-language bridge for clinical report generation

## 📁 Repository Structure

```
retina_gpt/
│
├── 📂 data/                        # Data layer
│   ├── raw/                        # Raw fundus images (unprocessed)
│   ├── processed/                  # Preprocessed & cached tensors
│   └── datasets/                   # Dataset loader modules
│       ├── __init__.py
│       ├── fundus_dataset.py       # Base fundus dataset class
│       ├── eyepacs_dataset.py      # EyePACS DR grading dataset
│       └── drive_dataset.py        # DRIVE vessel segmentation dataset
│
├── 📂 models/                      # Model layer
│   ├── __init__.py
│   ├── backbone/                   # Vision encoder
│   │   ├── __init__.py
│   │   ├── retina_vit.py           # Retina Vision Transformer backbone
│   │   └── patch_embed.py          # Custom fundus-aware patch embedding
│   ├── heads/                      # Task-specific heads
│   │   ├── __init__.py
│   │   ├── quality_head.py         # Image quality assessment head
│   │   ├── classification_head.py  # DR grading classification head
│   │   ├── segmentation_head.py    # Vessel segmentation decoder head
│   │   └── detection_head.py       # Lesion detection head
│   └── language/                   # Vision-language bridge
│       ├── __init__.py
│       ├── report_generator.py     # Medical report generation model
│       └── prompt_templates.py     # Clinical prompt engineering templates
│
├── 📂 training/                    # Training layer
│   ├── __init__.py
│   ├── trainer.py                  # Main trainer class with full loop
│   ├── losses.py                   # Task-specific loss functions
│   ├── metrics.py                  # Evaluation metrics (AUC, Dice, mAP)
│   ├── schedulers.py               # LR schedulers & warmup strategies
│   └── callbacks.py                # Training callbacks & checkpointing
│
├── 📂 inference/                   # Inference layer
│   ├── __init__.py
│   ├── pipeline.py                 # End-to-end inference pipeline
│   └── postprocessing.py           # Output postprocessing & formatting
│
├── 📂 api/                         # REST API layer
│   ├── __init__.py
│   ├── main.py                     # FastAPI application entrypoint
│   ├── routes.py                   # API route definitions
│   ├── schemas.py                  # Pydantic request/response schemas
│   └── middleware.py               # Auth, logging, rate limiting
│
├── 📂 configs/                     # Configuration layer
│   ├── model_config.yaml           # Model architecture hyperparameters
│   ├── training_config.yaml        # Training hyperparameters
│   └── inference_config.yaml       # Inference settings
│
├── 📂 utils/                       # Utilities layer
│   ├── __init__.py
│   ├── preprocessing.py            # Retinal image preprocessing
│   ├── augmentation.py             # Medical imaging augmentations
│   ├── visualization.py            # Result visualization tools
│   └── logging_utils.py            # Experiment logging (W&B, TensorBoard)
│
├── 📂 tests/                       # Test suite
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── 📂 scripts/                     # Utility scripts
│   ├── train.py                    # Training entrypoint script
│   ├── evaluate.py                 # Evaluation script
│   └── export_model.py             # ONNX/TorchScript export
│
├── 📂 notebooks/                   # Research notebooks
│   └── exploration.ipynb
│
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Train
python scripts/train.py --config configs/training_config.yaml

# Inference
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --image path/to/fundus.jpg

# API Server
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## 🧪 Supported Tasks

| Task | Head | Metric |
|------|------|--------|
| Image Quality Assessment | QualityHead | Accuracy |
| DR Classification (5-class) | ClassificationHead | AUC / Kappa |
| Vessel Segmentation | SegmentationHead | Dice / IoU |
| Lesion Detection | DetectionHead | mAP |
| Clinical Report Generation | ReportGenerator | BLEU / CIDEr |

## 📄 License
MIT License — For research use. Not for clinical deployment without validation.
