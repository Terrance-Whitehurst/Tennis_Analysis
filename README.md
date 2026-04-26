# Tennis Analysis

![Tennis Analysis Demo](reports/figures/demo.gif)

End-to-end tennis video analysis combining three deep learning models — ball tracking, player detection, and court keypoint detection — with a unified visualization pipeline.

- **Ball Tracking**: RF-DETR for per-frame tennis ball detection with fading trajectory trail visualization
- **Player Detection**: RF-DETR (Detection Transformer) for real-time player bounding boxes with ByteTrack temporal smoothing
- **Court Detection**: YOLO-Pose for 14-keypoint court geometry estimation with skeleton wireframe overlay

## Project Structure

This project follows the [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) (CCDS v2) convention.

```
Tennis_Analysis/
├── Makefile                  # Convenience commands (make train, make test, etc.)
├── README.md                 # This file
├── pyproject.toml            # Package metadata and dependencies
├── requirements.txt          # Flat dependency list
│
├── configs/                  # Experiment configs
│   └── court_keypoint.yaml   # YOLO dataset config for court keypoints
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
│   ├── ball_detection/       # RF-DETR checkpoints from SageMaker
│   ├── player_detection/     # RF-DETR checkpoints from SageMaker
│   └── court_keypoint/       # YOLO-Pose weights from SageMaker
│
├── notebooks/                # Jupyter notebooks for exploration
│                             # Naming: <number>-<initials>-<description>.ipynb
│
├── experiments/              # Experiment outputs, TensorBoard logs, results
│
├── src/                      # Main Python package (installable via pip install -e .)
│   ├── datasets/             # Data loading and preprocessing
│   ├── models/               # Model architecture definitions
│   ├── training/             # Training loops and schedulers
│   │   ├── train_ball_detection.py         # RF-DETR
│   │   ├── train_player_detection.py       # RF-DETR
│   │   └── train_court_keypoint.py         # YOLO-Pose
│   ├── inference/            # Inference and prediction pipelines
│   │   └── ball_tracking.py  # End-to-end RF-DETR ball tracking
│   └── utils/                # Shared helpers and visualization
│       └── general.py
│
├── scripts/                  # Standalone CLI scripts (not part of the package)
│   ├── convert_coco_to_yolo_kpt.py  # Dataset format converter
│   ├── test_models_on_video.py      # Run all models on test video
│   └── sagemaker/            # AWS SageMaker training launchers
│       ├── entry_player_detection.py
│       ├── entry_court_keypoint.py
│       ├── entry_ball_detection.py
│       ├── launch_player_detection.py
│       ├── launch_court_keypoint.py
│       └── launch_ball_detection.py
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
```

## Usage

Use `make help` to see all available commands.

### Test All Models on Video

```bash
# Run all models (ball tracking + player detection + court keypoint)
make test-models

# Or directly:
python scripts/test_models_on_video.py --video data/raw/test_video/Test_Clip_1.mp4

# Selectively disable models:
python scripts/test_models_on_video.py --no-ball     # skip ball tracking
python scripts/test_models_on_video.py --no-court    # skip court keypoint model
python scripts/test_models_on_video.py --no-player   # skip player detection model

# Custom model paths:
python scripts/test_models_on_video.py \
    --ball-model models/ball_detection/checkpoint_best_total.pth
```

Output goes to `reports/figures/` (annotated video + sample frames). Ball tracking runs per-frame with RF-DETR and overlays a fading trajectory trail alongside the other model annotations.

### Inference (Ball Tracking)

```bash
# Default — runs on test video with default checkpoint path
python -m src.inference.ball_tracking

# Custom paths:
python -m src.inference.ball_tracking \
    --video data/raw/test_video/Test_Clip_1.mp4 \
    --rfdetr-model models/ball_detection/checkpoint_best_total.pth \
    --output reports/figures/ball_tracking.mp4
```

### Training — Ball Detection (RF-DETR)

```bash
# Train with base model (default)
python -m src.training.train_ball_detection

# Train with large model for better accuracy
python -m src.training.train_ball_detection \
    --model large \
    --epochs 100 \
    --batch_size 4
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

## Models

### Ball Tracking — RF-DETR

RF-DETR fine-tuned to detect tennis balls in broadcast video footage. Runs per-frame —
no temporal stacking. The highest-confidence detection per frame is taken as "the ball";
frames below threshold are marked as no-ball. A fading trajectory trail is drawn using
supervision's TraceAnnotator.

Uses the same base/large architecture as player detection. Dataset is COCO-format
with tennis ball bounding boxes.

### Player Detection — RF-DETR

Real-time Detection Transformer from Roboflow. Accepts COCO-format datasets natively.
Available in two sizes:
- **RFDETRBase**: Faster training and inference
- **RFDETRLarge**: Higher accuracy

### Court Detection — YOLO-Pose

Ultralytics YOLO pose estimation model fine-tuned for tennis court keypoint detection.
Detects 14 keypoints that define the court geometry (corners, service lines, center marks)
with a skeleton graph connecting them.

## Training on AWS SageMaker

Player detection, court keypoint, and ball detection training can all be launched as SageMaker training jobs on GPU instances. The launcher scripts handle data upload to S3, job configuration, and submission.

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

### Instance Type Recommendations

| Task | Budget | Recommended Instance | GPU | VRAM |
|------|--------|---------------------|-----|------|
| Player Detection (base) | Low | `ml.g4dn.xlarge` | T4 | 16 GB |
| Player Detection (large) | Medium | `ml.g5.xlarge` | A10G | 24 GB |
| Court Keypoint (640px) | Low | `ml.g4dn.xlarge` | T4 | 16 GB |
| Court Keypoint (1280px) | Medium | `ml.g5.xlarge` | A10G | 24 GB |
| Ball Detection (base) | Low | `ml.g4dn.xlarge` | T4 | 16 GB |
| Ball Detection (large) | Medium | `ml.g5.xlarge` | A10G | 24 GB |
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
tar xzf models/player_detection/model.tar.gz -C models/player_detection/
tar xzf models/court_keypoint/model.tar.gz -C models/court_keypoint/
```

Model checkpoints are stored in `models/` (gitignored). Architecture definitions live in `src/models/`.

## References

- [RF-DETR](https://github.com/roboflow/rf-detr) — Real-time Detection Transformer
- [Ultralytics YOLO](https://docs.ultralytics.com/) — YOLO-Pose for keypoint detection
