"""
Train Ultralytics YOLO-Pose model for tennis court keypoint detection.

Uses YOLO's built-in keypoint/pose estimation to detect 14 court keypoints
that define the tennis court geometry. The model learns both the bounding box
of the court and the precise (x, y) location of each keypoint.

Prerequisites:
    1. Convert COCO annotations to YOLO format first:
       python scripts/convert_coco_to_yolo_kpt.py

    2. This creates data/processed/court_keypoint_yolo/ with YOLO-format labels

Dataset:
    - 828 training images (1280x720)
    - 55 validation images
    - 37 test images
    - 14 keypoints per court with skeleton connections
    - 1 class: tenniscourt

Usage:
    python -m src.training.train_court_keypoint
    python -m src.training.train_court_keypoint --model yolo11m-pose.pt --epochs 100 --imgsz 1280
    python -m src.training.train_court_keypoint --resume experiments/court_keypoint/weights/last.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(description="Train YOLO-Pose for court keypoint detection")

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m-pose.pt",
        help="Pretrained YOLO pose model to fine-tune (e.g., yolo11n-pose.pt, yolo11s-pose.pt, yolo11m-pose.pt, yolo11l-pose.pt, yolo11x-pose.pt)",
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="configs/court_keypoint.yaml",
        help="Path to dataset YAML config",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor (lr0 * lrf)")
    parser.add_argument("--momentum", type=float, default=0.937, help="SGD momentum / Adam beta1")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (0 to disable)")
    parser.add_argument("--optimizer", type=str, default="auto", choices=["SGD", "Adam", "AdamW", "auto"], help="Optimizer")

    # Augmentation
    parser.add_argument("--hsv_h", type=float, default=0.015, help="HSV hue augmentation")
    parser.add_argument("--hsv_s", type=float, default=0.7, help="HSV saturation augmentation")
    parser.add_argument("--hsv_v", type=float, default=0.4, help="HSV value augmentation")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation degrees")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation augmentation")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Horizontal flip probability (0.0 for court — flipping breaks keypoint order)")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability")

    # Loss weights
    parser.add_argument("--pose", type=float, default=12.0, help="Pose/keypoint loss weight")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss weight")
    parser.add_argument("--cls", type=float, default=0.5, help="Classification loss weight")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss weight")

    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="experiments",
        help="Project directory for saving runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="court_keypoint",
        help="Run name within project directory",
    )

    # Hardware
    parser.add_argument("--device", type=str, default=None, help="Device (0, 0,1, cpu, mps). Auto-detected if not set.")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")

    return parser.parse_args()


def ensure_dataset_converted(data_yaml_path):
    """Check if YOLO-format dataset exists, run converter if not."""
    import yaml

    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    dataset_path = Path(cfg["path"])
    train_images = dataset_path / cfg["train"]

    if not train_images.exists():
        print(f"YOLO dataset not found at {dataset_path}")
        print("Running COCO-to-YOLO converter...")
        from scripts.convert_coco_to_yolo_kpt import main as convert_main
        sys.argv = ["convert_coco_to_yolo_kpt.py"]  # Reset argv for converter's argparse
        convert_main()
        print()

    assert train_images.exists(), \
        f"Training images not found at {train_images}. Run: python scripts/convert_coco_to_yolo_kpt.py"


def train(args):
    # Validate config exists
    data_path = Path(args.data)
    assert data_path.exists(), f"Dataset config not found: {data_path}"

    # Auto-convert if needed
    ensure_dataset_converted(data_path)

    # Handle resume vs new training
    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    # Initialize model from pretrained
    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    # Build training kwargs
    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        optimizer=args.optimizer,
        # Augmentation
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        # Loss weights
        pose=args.pose,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        # Output
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Hardware
        workers=args.workers,
        # Logging
        verbose=True,
        plots=True,
    )

    if args.device is not None:
        train_kwargs["device"] = args.device

    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Image size: {args.imgsz}")
    print(f"LR: {args.lr0} -> {args.lr0 * args.lrf}, Optimizer: {args.optimizer}")
    print(f"Pose loss weight: {args.pose}, Patience: {args.patience}")
    print(f"Output: {args.project}/{args.name}")
    print()

    # Train
    results = model.train(**train_kwargs)

    # Copy best weights to models/ for easy access
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        dest = Path("models") / "court_keypoint_yolo_best.pt"
        os.makedirs(dest.parent, exist_ok=True)
        import shutil
        shutil.copy2(best_weights, dest)
        print(f"\nBest weights copied to {dest}")

    # Run validation on best model
    print("\nRunning final validation...")
    best_model = YOLO(str(best_weights)) if best_weights.exists() else model
    val_results = best_model.val(data=str(data_path))

    print("\n=== Training Complete ===")
    print(f"Best weights: {best_weights}")
    print(f"Results saved to: {args.project}/{args.name}")

    return results


if __name__ == "__main__":
    args = get_args()
    train(args)
