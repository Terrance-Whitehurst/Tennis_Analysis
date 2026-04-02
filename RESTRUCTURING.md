# Repository Restructuring Guide

This document explains the full restructuring of the Tennis Analysis repository from a flat research-style layout into a production-quality ML project.

---

## Why Restructure?

The original layout had all code nested inside `Models/TrackNetV3/` as a monolithic block — model definitions, training loops, evaluation, inference, data loading, and interactive tools were all siblings in one directory. This makes it hard to:

- **Reuse components** (e.g., importing the model architecture without pulling in training code)
- **Test in isolation** (no clear boundaries between modules)
- **Onboard new contributors** (unclear where to find or add things)
- **Scale the project** (adding a second model or dataset requires rethinking the layout)

The new structure follows the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) convention, which is the de facto standard for ML repositories in production.

---

## Before vs After

### Original Structure

```
Tennis_Analysis/
├── ball_tracking.py                    # Standalone inference script
├── Test_Video/Test_Clip_1.mp4
├── Datasets/
│   ├── Player_Detection/
│   └── Tennis_Court_Keypoint/
└── Models/TrackNetV3/                  # Everything in one flat directory
    ├── model.py                        # Architectures
    ├── dataset.py                      # Data loading
    ├── train.py                        # Training
    ├── test.py                         # Evaluation
    ├── predict.py                      # Inference
    ├── preprocess.py                   # Data prep
    ├── correct_label.py               # Interactive UI
    ├── error_analysis.py              # Dashboard
    ├── generate_mask_data.py          # Mask generation
    ├── utils/{general,metric,visualize}.py
    ├── ckpts/*.pt
    └── corrected_test_label/
```

### New Structure

```
Tennis_Analysis/
├── configs/                    # Experiment configs and label corrections
│   ├── drop_frame.json
│   └── corrected_test_label/
│
├── data/                       # All data lives here (not committed to git)
│   ├── raw/                    # Original datasets + test videos
│   │   ├── Player_Detection/
│   │   ├── Tennis_Court_Keypoint/
│   │   ├── Artifacts/
│   │   └── test_video/
│   ├── interim/                # Intermediate processing outputs
│   └── processed/              # Final processed data ready for training
│
├── experiments/                # Experiment logs, TensorBoard runs, results
│
├── models/                     # Trained model checkpoints
│   ├── TrackNet_best.pt
│   └── InpaintNet_best.pt
│
├── notebooks/                  # Jupyter notebooks for exploration
│
├── src/                        # Main Python package (installable)
│   ├── __init__.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── tracknet_dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── tracknet.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_tracknet.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── ball_tracking.py
│   │   └── predict.py
│   └── utils/
│       ├── __init__.py
│       ├── general.py
│       ├── metric.py
│       └── visualize.py
│
├── scripts/                    # Runnable CLI scripts (not part of the package)
│   ├── preprocess.py
│   ├── correct_label.py
│   ├── error_analysis.py
│   └── generate_mask_data.py
│
├── tests/                      # Unit tests
│
├── pyproject.toml              # Package definition + dependencies
├── requirements.txt            # Flat dependency list
├── .gitignore
└── README.md
```

---

## Design Decisions

### 1. `src/` as the Main Package

All reusable Python code lives under `src/` with proper `__init__.py` files, making it an installable Python package. This means:

- You can `pip install -e .` (or `uv pip install -e .`) and import from anywhere
- No more `sys.path` hacks in production code
- Clear dependency graph between modules

The `pyproject.toml` uses [setuptools package discovery](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html) to find everything under `src/`.

### 2. Separation by Responsibility

Each subdirectory under `src/` has a single, clear purpose:

| Directory | Responsibility | Original File |
|---|---|---|
| `src/models/` | Neural network architecture definitions only | `model.py` |
| `src/datasets/` | Data loading, preprocessing, and augmentation | `dataset.py` |
| `src/training/` | Training loops, optimizers, schedulers | `train.py` |
| `src/evaluation/` | Metrics computation, model evaluation, testing | `test.py` |
| `src/inference/` | Production inference pipelines | `ball_tracking.py`, `predict.py` |
| `src/utils/` | Shared helpers, constants, visualization | `utils/` |

This follows the [Single Responsibility Principle](https://en.wikipedia.org/wiki/Single-responsibility_principle). A change to the model architecture doesn't touch training code. A change to evaluation metrics doesn't touch inference.

### 3. Scripts vs Package Code

**Scripts** (`scripts/`) are entry points meant to be run directly from the command line. They contain argument parsing, setup logic, and orchestration — but delegate actual work to `src/`.

**Package code** (`src/`) contains reusable, importable logic with no side effects at import time.

This distinction matters because:
- Scripts can be swapped out (e.g., replace argparse with Hydra) without touching core logic
- Package code can be imported in notebooks, tests, or other scripts
- Interactive tools (Dash apps) live in `scripts/` because they're not importable library code

Each script includes a `sys.path.insert(0, project_root)` line so it can be run directly with `python scripts/foo.py` without installing the package first.

### 4. Data Directory Convention

Following [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#directory-structure):

- **`data/raw/`** — Immutable original data. Never modified after initial placement.
- **`data/interim/`** — Intermediate transformations (e.g., extracted frames, computed medians).
- **`data/processed/`** — Final datasets ready for model consumption.

All of `data/` is in `.gitignore` because datasets are too large for version control. Use [DVC](https://dvc.org/) or cloud storage for data versioning.

### 5. Configs Separated from Code

Configuration files (`drop_frame.json`, corrected labels) moved to `configs/` instead of living alongside Python code. This makes it easy to:

- Swap experiment configurations
- Version control configs without touching code
- Eventually migrate to [Hydra](https://hydra.cc/) for structured config management

### 6. Model Checkpoints in `models/`

Checkpoint files (`.pt`) moved to a dedicated `models/` directory at the project root, separate from model architecture code (`src/models/`). This avoids confusion between:

- **Model definitions** (Python classes) → `src/models/`
- **Model weights** (serialized tensors) → `models/`

---

## Import Changes

The biggest code change was rewriting all internal imports from relative/flat style to absolute package imports.

### Before (flat relative imports)

```python
# In train.py — relies on all files being in the same directory
from model import TrackNet, InpaintNet
from dataset import Shuttlecock_Trajectory_Dataset
from test import eval_tracknet, eval_inpaintnet
from utils.general import get_model
from utils.metric import WBCELoss
```

### After (absolute package imports)

```python
# In src/training/train_tracknet.py — explicit, unambiguous paths
from src.models.tracknet import TrackNet, InpaintNet
from src.datasets.tracknet_dataset import Shuttlecock_Trajectory_Dataset
from src.evaluation.evaluate import eval_tracknet, eval_inpaintnet
from src.utils.general import get_model
from src.utils.metric import WBCELoss
```

### Why Absolute Imports?

- **Unambiguous**: `from src.models.tracknet import TrackNet` can only mean one thing
- **Refactor-safe**: Moving a file won't silently break imports in other files
- **IDE-friendly**: Autocomplete and "go to definition" work correctly
- **PEP 328 compliant**: [Absolute imports are the recommended default](https://peps.python.org/pep-0328/)

### The `sys.path` Hack in ball_tracking.py

The original `ball_tracking.py` had this pattern:

```python
TRACKNET_DIR = os.path.join(os.path.dirname(__file__), "TrackNetV3")
sys.path.insert(0, TRACKNET_DIR)
from model import TrackNet, InpaintNet
```

This is a common workaround when code isn't structured as a package. It's fragile (breaks if you move the file) and pollutes `sys.path`. The new version simply does:

```python
from src.models.tracknet import TrackNet, InpaintNet
```

---

## Complete File Migration Map

| Original Path | New Path | Changes |
|---|---|---|
| `ball_tracking.py` | `src/inference/ball_tracking.py` | Removed `sys.path` hack, updated model import, updated default paths to new locations |
| `Models/TrackNetV3/model.py` | `src/models/tracknet.py` | No changes (only external imports) |
| `Models/TrackNetV3/dataset.py` | `src/datasets/tracknet_dataset.py` | `from utils.general import ...` → `from src.utils.general import ...` |
| `Models/TrackNetV3/train.py` | `src/training/train_tracknet.py` | 5 import lines updated to `src.*` |
| `Models/TrackNetV3/test.py` | `src/evaluation/evaluate.py` | 3 import lines updated to `src.*` |
| `Models/TrackNetV3/predict.py` | `src/inference/predict.py` | 3 import lines updated to `src.*` |
| `Models/TrackNetV3/utils/general.py` | `src/utils/general.py` | `from model import ...` → `from src.models.tracknet import ...` |
| `Models/TrackNetV3/utils/metric.py` | `src/utils/metric.py` | No changes (only external imports) |
| `Models/TrackNetV3/utils/visualize.py` | `src/utils/visualize.py` | 2 import lines updated to `src.*` |
| `Models/TrackNetV3/preprocess.py` | `scripts/preprocess.py` | Added `sys.path` setup, 3 imports updated, paths to corrected labels updated |
| `Models/TrackNetV3/correct_label.py` | `scripts/correct_label.py` | Added `sys.path` setup, 2 imports updated |
| `Models/TrackNetV3/error_analysis.py` | `scripts/error_analysis.py` | Added `sys.path` setup, 2 imports updated |
| `Models/TrackNetV3/generate_mask_data.py` | `scripts/generate_mask_data.py` | Added `sys.path` setup, 2 imports updated |
| `Datasets/*` | `data/raw/` | Moved as-is |
| `Test_Video/*` | `data/raw/test_video/` | Moved as-is |
| `Models/TrackNetV3/ckpts/*.pt` | `models/` | Moved as-is |
| `Models/TrackNetV3/corrected_test_label/` | `configs/corrected_test_label/` | Moved as-is |
| `Models/TrackNetV3/requirements.txt` | `requirements.txt` | Updated with additional dependencies (tqdm, tensorboard, pycocotools, matplotlib) |

---

## New Files Created

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, and build configuration |
| `README.md` | Project documentation with usage examples |
| `.gitignore` | Excludes data, checkpoints, caches, and IDE files |
| `src/__init__.py` | Makes `src` an importable package |
| `src/*/\__init__.py` | Makes each subdirectory an importable subpackage |
| `data/*/.gitkeep` | Preserves empty directories in git |

---

## How to Run After Restructuring

### Install the package

```bash
# Using uv (recommended)
uv pip install -e .

# Or standard pip
pip install -e .
```

### Run training

```bash
python -m src.training.train_tracknet \
    --model_name TrackNet \
    --seq_len 8 \
    --epochs 30 \
    --batch_size 10 \
    --bg_mode concat \
    --save_dir experiments/tracknet_v1
```

### Run evaluation

```bash
python -m src.evaluation.evaluate \
    --tracknet_file models/TrackNet_best.pt \
    --split test \
    --eval_mode weight
```

### Run inference

```bash
python -m src.inference.ball_tracking
```

### Run scripts

```bash
python scripts/preprocess.py
python scripts/correct_label.py --split test
python scripts/error_analysis.py --split test
python scripts/generate_mask_data.py --tracknet_file models/TrackNet_best.pt
```

---

## References

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) — The project template this layout is based on
- [PEP 328 — Imports: Multi-Line and Absolute/Relative](https://peps.python.org/pep-0328/) — Why absolute imports are preferred
- [Setuptools Package Discovery](https://setuptools.pypa.io/en/latest/userguide/package_discovery.html) — How `pyproject.toml` finds the `src` package
- [DVC (Data Version Control)](https://dvc.org/) — Recommended for versioning large datasets
- [Hydra](https://hydra.cc/) — Framework for managing experiment configurations
- [PyTorch Project Template](https://github.com/victoresque/pytorch-template) — Another common ML project layout
- [Lightning CLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) — Alternative for structured training scripts
