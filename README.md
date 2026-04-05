# Tennis Analysis

![Tennis Analysis Demo](reports/figures/demo.gif)

End-to-end tennis video analysis combining four deep learning models — ball tracking, player detection, court keypoint detection, and scoreboard detection — with a unified visualization pipeline.

- **Ball Tracking**: TrackNetV3 (2D U-Net) for heatmap-based ball detection + InpaintNet (1D U-Net) for trajectory gap filling, with fading trajectory trail visualization
- **Player Detection**: RF-DETR (Detection Transformer) for real-time player bounding boxes with ByteTrack temporal smoothing
- **Court Detection**: YOLO-Pose for 14-keypoint court geometry estimation with skeleton wireframe overlay
- **Scoreboard Detection**: RF-DETR for scoreboard bounding box detection

## Project Structure

This project follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) (CCDS v2) convention.

```
Tennis_Analysis/
├── Makefile                  # Convenience commands (make train, make test, etc.)
├── README.md                 # This file
├── pyproject.toml            # Package metadata and dependencies
├── requirements.txt          # Flat dependency list
│
├── configs/                  # Experiment configs and label corrections
│   ├── drop_frame.json
│   ├── court_keypoint.yaml   # YOLO dataset config for court keypoints
│   └── corrected_test_label/
│
├── data/                     # Dataset storage — NOT committed to git
│   ├── raw/                  # Original immutable datasets and test videos
│   ├── interim/              # Intermediate transformed data
│   ├── processed/            # Final canonical data for modeling
│   └── external/             # Third-party data sources
│
├── docs/                     # Project documentation
├── references/               # Data dictionaries, manuals, explanatory materials
├── reports/                  # Generated analysis outputs
│   └── figures/              # Generated graphics and annotated frames
│
├── models/                   # Trained model checkpoints (not committed)
│   ├── TrackNet_best.pt      # TrackNetV3 pretrained weights
│   ├── InpaintNet_best.pt    # InpaintNet pretrained weights
│   ├── player_detection/     # RF-DETR checkpoints from SageMaker
│   ├── court_keypoint/       # YOLO-Pose weights from SageMaker
│   └── scoreboard_detection/ # RF-DETR checkpoints from SageMaker
│
├── notebooks/                # Jupyter notebooks for exploration
│                             # Naming: <number>-<initials>-<description>.ipynb
│
├── experiments/              # Experiment outputs, TensorBoard logs, results
│
├── src/                      # Main Python package (installable via pip install -e .)
│   ├── datasets/             # Data loading and preprocessing
│   │   └── tracknet_dataset.py
│   ├── models/               # Model architecture definitions
│   │   └── tracknet.py       # TrackNet + InpaintNet
│   ├── training/             # Training loops and schedulers
│   │   ├── train_tracknet.py
│   │   ├── train_player_detection.py       # RF-DETR
│   │   ├── train_court_keypoint.py         # YOLO-Pose
│   │   └── train_scoreboard_detection.py   # RF-DETR
│   ├── evaluation/           # Metrics and evaluation pipelines
│   │   └── evaluate.py
│   ├── inference/            # Inference and prediction pipelines
│   │   ├── ball_tracking.py  # End-to-end video ball tracking
│   │   └── predict.py        # Batch prediction
│   └── utils/                # Shared helpers, constants, visualization
│       ├── general.py
│       ├── metric.py
│       └── visualize.py
│
├── scripts/                  # Standalone CLI scripts (not part of the package)
│   ├── preprocess.py         # Data preprocessing
│   ├── correct_label.py      # Interactive label correction UI
│   ├── error_analysis.py     # Error analysis dashboard
│   ├── generate_mask_data.py # Generate inpainting masks
│   ├── convert_coco_to_yolo_kpt.py  # Dataset format converter
│   ├── test_models_on_video.py      # Run all models on test video
│   └── sagemaker/            # AWS SageMaker training launchers
│       ├── entry_player_detection.py
│       ├── entry_court_keypoint.py
│       ├── entry_scoreboard_detection.py
│       ├── launch_player_detection.py
│       ├── launch_court_keypoint.py
│       └── launch_scoreboard_detection.py
│
└── tests/                    # Unit and integration tests
```

## Setup

```bash
# Install dependencies
uv pip install -e .

# Or with requirements.txt
uv pip install -r requirements.txt

# Pull trained models from S3 (requires AWS credentials)
make pull-models

# Download pretrained TrackNetV3 weights (from official repo via Google Drive)
pip install gdown
gdown "https://drive.google.com/uc?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA" -O models/TrackNetV3_ckpts.zip
unzip models/TrackNetV3_ckpts.zip -d models/ && mv models/ckpts/* models/ && rmdir models/ckpts
rm models/TrackNetV3_ckpts.zip
```

## Usage

Use `make help` to see all available commands.

### Test All Models on Video

```bash
# Run all models (ball tracking + player detection + court keypoint + scoreboard detection)
make test-models

# Or directly:
python scripts/test_models_on_video.py --video data/raw/test_video/Test_Clip_1.mp4

# Selectively disable models:
python scripts/test_models_on_video.py --no-ball         # skip ball tracking
python scripts/test_models_on_video.py --no-court        # skip court keypoint model
python scripts/test_models_on_video.py --no-player       # skip player detection model
python scripts/test_models_on_video.py --no-scoreboard   # skip scoreboard detection model

# Custom model paths:
python scripts/test_models_on_video.py \
    --tracknet-model models/TrackNet_best.pt \
    --inpaintnet-model models/InpaintNet_best.pt
```

Output goes to `reports/figures/` (annotated video + sample frames). Ball tracking runs as a pre-processing step (TrackNetV3 processes 8-frame sequences with temporal ensemble), then overlays a fading trajectory trail on each frame alongside the other model annotations.

### Inference (Ball Tracking)

```bash
python -m src.inference.ball_tracking
```

### Training — Ball Tracking (TrackNet)

```bash
python -m src.training.train_tracknet \
    --model_name TrackNet \
    --seq_len 8 \
    --epochs 30 \
    --batch_size 10 \
    --bg_mode concat \
    --save_dir experiments/tracknet_v1
```

### Training — Player Detection (RF-DETR)

```bash
# Train with base model (default)
python -m src.training.train_player_detection

# Train with large model for better accuracy
python -m src.training.train_player_detection \
    --model large \
    --epochs 100 \
    --batch_size 4 \
    --image_size 560

# Resume from checkpoint
python -m src.training.train_player_detection \
    --resume experiments/player_detection/best_checkpoint.pt
```

Dataset: `data/raw/Player_Detection/` (COCO format, 5166 train / 570 val / 14 test images)
Classes: `player-back`, `player-front`

### Training — Court Keypoint Detection (YOLO-Pose)

```bash
# Step 1: Convert COCO annotations to YOLO format (only needed once)
python scripts/convert_coco_to_yolo_kpt.py

# Step 2: Train (auto-converts if not done yet)
python -m src.training.train_court_keypoint

# Train with larger model and native resolution
python -m src.training.train_court_keypoint \
    --model yolo11l-pose.pt \
    --epochs 200 \
    --imgsz 1280 \
    --batch 8

# Resume from checkpoint
python -m src.training.train_court_keypoint \
    --resume experiments/court_keypoint/weights/last.pt
```

Dataset: `data/raw/Tennis_Court_Keypoint/` (828 train / 55 val / 37 test images)
Detects 14 keypoints defining the court geometry with skeleton connections.

### Training — Scoreboard Detection (RF-DETR)

```bash
# Train with base model (default)
python -m src.training.train_scoreboard_detection

# Train with large model for better accuracy
python -m src.training.train_scoreboard_detection \
    --model large \
    --epochs 100 \
    --batch_size 4

# Resume from checkpoint
python -m src.training.train_scoreboard_detection \
    --resume models/scoreboard_detection/checkpoint.pt
```

Dataset: `data/raw/Scoreboard_Detection/` (COCO format, 161 train images)
Classes: `scoreboard`

### Evaluation

```bash
python -m src.evaluation.evaluate \
    --tracknet_file models/TrackNet_best.pt \
    --split test \
    --eval_mode weight
```

### Data Preprocessing

```bash
python scripts/preprocess.py
```

### Interactive Label Correction

```bash
python scripts/correct_label.py --split test
```

### Error Analysis Dashboard

```bash
python scripts/error_analysis.py --split test
```

## Models

### Ball Tracking — TrackNetV3 + InpaintNet

Pretrained weights from the [official TrackNetV3 repo](https://github.com/qaz812345/TrackNetV3), originally trained on badminton shuttlecock data but generalizes well to tennis balls (small, fast-moving objects on a court).

- **TrackNetV3**: 2D U-Net encoder-decoder
  - Input: (N, 27, 288, 512) — 8 RGB frames + 1 median background, channel-concatenated
  - Output: (N, 8, 288, 512) — per-frame heatmaps via sigmoid
  - Temporal ensemble with triangular weighting for overlapping sequence predictions

- **InpaintNet**: 1D U-Net operating on coordinate sequences
  - Input: (N, L, 3) — normalized (x, y) + inpaint mask
  - Output: (N, L, 2) — refined (x, y) coordinates
  - Fills in trajectory gaps where the ball was occluded or missed by TrackNet

### Player Detection — RF-DETR

Real-time Detection Transformer from Roboflow. Accepts COCO-format datasets natively.
Available in two sizes:
- **RFDETRBase**: Faster training and inference
- **RFDETRLarge**: Higher accuracy

### Court Detection — YOLO-Pose

Ultralytics YOLO pose estimation model fine-tuned for tennis court keypoint detection.
Detects 14 keypoints that define the court geometry (corners, service lines, center marks)
with a skeleton graph connecting them.

### Scoreboard Detection — RF-DETR

RF-DETR fine-tuned to detect scoreboard overlays in broadcast tennis footage.
Uses the same base/large architecture as player detection. Dataset is COCO-format
with a single `scoreboard` class.

## Training on AWS SageMaker

Player detection, court keypoint, and scoreboard detection training can all be launched as SageMaker training jobs on GPU instances. The launcher scripts handle data upload to S3, job configuration, and submission.

### Prerequisites

```bash
# Install SageMaker SDK
uv pip install sagemaker boto3

# Configure AWS credentials
aws configure
```

You'll need a SageMaker execution role ARN with S3 access. Create one in the IAM console or use:
```bash
ROLE_ARN="arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_ROLE>"
```

### Launch Player Detection (RF-DETR) on SageMaker

```bash
# Basic — uploads local data, starts training on ml.g4dn.xlarge
python scripts/sagemaker/launch_player_detection.py \
    --role $ROLE_ARN

# Production — larger model, bigger instance, spot pricing
python scripts/sagemaker/launch_player_detection.py \
    --role $ROLE_ARN \
    --instance_type ml.g5.xlarge \
    --model large \
    --epochs 100 \
    --batch_size 16 \
    --spot

# Data already on S3
python scripts/sagemaker/launch_player_detection.py \
    --role $ROLE_ARN \
    --s3_data s3://my-bucket/datasets/Player_Detection

# Wait for completion and stream logs
python scripts/sagemaker/launch_player_detection.py \
    --role $ROLE_ARN --wait
```

### Launch Court Keypoint (YOLO-Pose) on SageMaker

```bash
# Basic — uploads COCO data, converts to YOLO inside container, trains
python scripts/sagemaker/launch_court_keypoint.py \
    --role $ROLE_ARN

# Native resolution, larger model
python scripts/sagemaker/launch_court_keypoint.py \
    --role $ROLE_ARN \
    --instance_type ml.g5.xlarge \
    --model yolo11l-pose.pt \
    --imgsz 1280 \
    --epochs 200 \
    --batch 8 \
    --spot
```

### Launch Scoreboard Detection (RF-DETR) on SageMaker

```bash
# Basic — uploads local data, starts training on ml.g4dn.xlarge
python scripts/sagemaker/launch_scoreboard_detection.py \
    --role $ROLE_ARN

# Custom instance and hyperparams
python scripts/sagemaker/launch_scoreboard_detection.py \
    --role $ROLE_ARN \
    --instance_type ml.g5.xlarge \
    --model large \
    --epochs 100 \
    --batch_size 16

# Data already on S3
python scripts/sagemaker/launch_scoreboard_detection.py \
    --role $ROLE_ARN \
    --s3_data s3://my-bucket/datasets/Scoreboard_Detection
```

### Instance Type Recommendations

| Task | Budget | Recommended Instance | GPU | VRAM |
|------|--------|---------------------|-----|------|
| Player Detection (base) | Low | `ml.g4dn.xlarge` | T4 | 16 GB |
| Player Detection (large) | Medium | `ml.g5.xlarge` | A10G | 24 GB |
| Court Keypoint (640px) | Low | `ml.g4dn.xlarge` | T4 | 16 GB |
| Court Keypoint (1280px) | Medium | `ml.g5.xlarge` | A10G | 24 GB |
| Scoreboard Detection (base) | Low | `ml.g4dn.xlarge` | T4 | 16 GB |
| Scoreboard Detection (large) | Medium | `ml.g5.xlarge` | A10G | 24 GB |
| Any (fastest) | High | `ml.p3.2xlarge` | V100 | 16 GB |

Add `--spot` for ~60-70% cost savings (jobs may be interrupted and restarted).

### Retrieving Trained Models

After training completes, model artifacts are saved to S3. Use the Makefile target:
```bash
make pull-models
```

Or manually:
```bash
aws s3 cp s3://training-jobs-test-315109499400/tennis-analysis/models/player_detection/<job>/output/model.tar.gz models/player_detection/
aws s3 cp s3://training-jobs-test-315109499400/tennis-analysis/models/court_keypoint/<job>/output/model.tar.gz models/court_keypoint/
aws s3 cp s3://training-jobs-test-315109499400/tennis-analysis/models/scoreboard_detection/<job>/output/model.tar.gz models/scoreboard_detection/
tar xzf models/player_detection/model.tar.gz -C models/player_detection/
tar xzf models/court_keypoint/model.tar.gz -C models/court_keypoint/
tar xzf models/scoreboard_detection/model.tar.gz -C models/scoreboard_detection/
```

Model checkpoints are stored in `models/` (gitignored). Architecture definitions live in `src/models/`.

## References

- [TrackNetV3](https://github.com/qaz812345/TrackNetV3) — "Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification" (ACM 2023)
- [RF-DETR](https://github.com/roboflow/rf-detr) — Real-time Detection Transformer
- [Ultralytics YOLO](https://docs.ultralytics.com/) — YOLO-Pose for keypoint detection
