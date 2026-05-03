"""
Train Ultralytics YOLO11-seg model for tennis court instance segmentation.

Segments three court region types from broadcast video frames:
    doubles_alley  (YOLO class 0) — side channels outside singles lines
    no_mans_land   (YOLO class 1) — area behind service box, inside baseline
    service_box    (YOLO class 2) — four boxes between net and service line

Prerequisites:
    1. Convert COCO annotations to YOLO-seg format first:
       python scripts/convert_coco_to_yolo_seg.py

    2. This creates data/processed/court_segmentation_yolo/ with YOLO-seg polygon labels.

Dataset:
    - 427 training images (854×480, Acapulco ATP broadcast)
    - 2 validation images ⚠️ (see WARNING below)
    - 3 active classes

WARNING — tiny validation set:
    The Roboflow export has only 2 validation images (both from the same video clip).
    mAP scores during training will be extremely noisy and should not be used to
    compare runs. The model will still train fine; just ignore the val metrics.
    To get meaningful val metrics, re-split before training — follow the re-split
    instructions printed at the end of scripts/convert_coco_to_yolo_seg.py.

─────────────────────────────────────────────
AUGMENTATION RATIONALE
─────────────────────────────────────────────
hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
    Broadcast court colors vary significantly by venue/lighting. Strong saturation
    and value jitter helps generalise across clay (red), hard (blue/green), and
    grass without touching hue aggressively (courts stay recognisably the same hue).

degrees=5.0
    Courts have rigid perspective geometry. Any rotation beyond ±5° would produce
    physically implausible views and confuse the model. Keep it small.

translate=0.1, scale=0.5
    Standard — allow the court to appear anywhere in the frame (broadcast cameras
    often re-crop). Scale 0.5 handles both tight crop and zoomed-out views.

shear=2.0
    Light shear simulates mild camera tilt without breaking court geometry.

perspective=0.0005
    Very slight perspective warp (near-zero) to improve robustness to different
    broadcast camera angles without distorting polygon labels badly.

flipud=0.0
    Vertical flip is physically impossible for a broadcast tennis camera. The net
    is always at mid-frame; flipping would produce an upside-down court. Hard off.

fliplr=0.5
    Tennis courts are bilaterally symmetric left-right. Horizontal flip doubles
    effective dataset size with zero label-quality cost. Always on.

mosaic=1.0
    With only 427 training images this is non-negotiable. Mosaic combines 4 frames
    per batch item, giving the model radically more spatial context per gradient
    step and preventing overfitting on repeated frames.

mixup=0.1
    Light alpha-blend between samples. Regularises boundaries at class edges
    (doubles alley / no-mans-land boundary) without being heavy enough to destroy
    polygon structure.

copy_paste=0.1
    Segmentation-specific augmentation: pastes segmented instances from other
    frames into the current image. Works well for structured scenes where region
    shapes are consistent across frames.

close_mosaic=10
    Disable mosaic in the final 10 epochs so the model can fine-tune on clean
    single-frame inputs before final evaluation.

cos_lr=True, patience=25, optimizer="auto"
    Cosine LR decay suits the ~100 epoch regime. auto-optimizer picks AdamW for
    YOLO11 which trains faster than SGD on small datasets.
─────────────────────────────────────────────

Usage:
    python -m src.training.train_court_segmentation
    python -m src.training.train_court_segmentation --model yolo11l-seg.pt --epochs 150
    python -m src.training.train_court_segmentation --resume experiments/court_segmentation/run/weights/last.pt
    python -m src.training.train_court_segmentation --imgsz 640 --batch 16 --device 0
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11-seg for tennis court instance segmentation"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m-seg.pt",
        help=(
            "Pretrained YOLO seg model to fine-tune. Options: "
            "yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, "
            "yolo11l-seg.pt, yolo11x-seg.pt"
        ),
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default="configs/court_segmentation.yaml",
        help="Path to dataset YAML config",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 for auto-batch based on available VRAM)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help=(
            "Input image size (default 640 — source images are 854×480, "
            "upscaling to 960 adds no information)"
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early stopping patience in epochs (0 to disable)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "AdamW", "auto"],
        help="Optimizer (auto selects AdamW for YOLO11)",
    )

    # Output
    parser.add_argument(
        "--project",
        type=str,
        default="experiments/court_segmentation",
        help="Project directory for saving runs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="run",
        help="Run name within project directory",
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on: 0, 0,1, cpu, mps. Auto-detected if empty.",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to last.pt checkpoint to resume interrupted training",
    )

    return parser.parse_args()


def ensure_dataset_converted(data_yaml_path: str) -> None:
    """Check if the YOLO-seg dataset exists; run the converter if not."""
    import yaml

    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve dataset path relative to the YAML file location
    yaml_dir = Path(data_yaml_path).resolve().parent
    dataset_path = (yaml_dir / cfg["path"]).resolve()
    train_images = dataset_path / cfg["train"]

    if not train_images.exists():
        print(f"YOLO-seg dataset not found at {dataset_path}")
        print("Running COCO-to-YOLO-seg converter automatically...")
        print()
        # Reset argv so the converter's argparse gets clean defaults
        saved_argv = sys.argv
        sys.argv = ["convert_coco_to_yolo_seg.py"]
        try:
            from scripts.convert_coco_to_yolo_seg import main as convert_main
            convert_main()
        finally:
            sys.argv = saved_argv
        print()

    if not train_images.exists():
        raise FileNotFoundError(
            f"Training images still not found at {train_images}. "
            "Run manually: python scripts/convert_coco_to_yolo_seg.py"
        )


def warn_if_small_val(data_yaml_path: str) -> None:
    """Warn the user if the validation split is suspiciously small."""
    import yaml

    with open(data_yaml_path) as f:
        cfg = yaml.safe_load(f)

    yaml_dir = Path(data_yaml_path).resolve().parent
    dataset_path = (yaml_dir / cfg["path"]).resolve()
    val_images = dataset_path / cfg["val"]

    if val_images.exists():
        n_val = len(list(val_images.glob("*")))
        if n_val < 10:
            print("=" * 60)
            print(f"⚠️  WARNING: val split has only {n_val} image(s).")
            print("   mAP and other val metrics will be extremely noisy.")
            print("   Training will proceed, but do NOT use val mAP to compare runs.")
            print()
            print("   To fix: carve ~55 images out of train into val.")
            print("   Run: python scripts/convert_coco_to_yolo_seg.py")
            print("   and follow the re-split instructions printed at the end.")
            print("=" * 60)
            print()


def train(args) -> None:
    # Validate config file exists
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    # Auto-convert COCO annotations if needed
    ensure_dataset_converted(str(data_path))

    # Warn about tiny val set
    warn_if_small_val(str(data_path))

    # ── Resume mode ─────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True)
        return

    # ── Fresh training ───────────────────────────────────────────────────────
    print(f"Loading pretrained model: {args.model}")
    model = YOLO(args.model)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        optimizer=args.optimizer,
        cos_lr=True,
        # ── Augmentations ────────────────────────────────────────────────────
        # Color: strong sat/val jitter for multi-venue generalisation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Geometric: keep light — court perspective geometry is rigid
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        # Flip: never vertical (broadcast camera), always horizontal (symmetric court)
        flipud=0.0,
        fliplr=0.5,
        # Composition augmentations: critical for 427-image dataset
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
        # ── Output ───────────────────────────────────────────────────────────
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # Only pass device if explicitly set (empty string = auto-detect)
    if args.device:
        train_kwargs["device"] = args.device

    print()
    print(f"Dataset:   {args.data}")
    print(f"Model:     {args.model}")
    print(f"Epochs:    {args.epochs}  |  Batch: {args.batch} (auto)  |  imgsz: {args.imgsz}")
    print(f"Optimizer: {args.optimizer}  |  cos_lr: True  |  Patience: {args.patience}")
    print(f"Output:    {args.project}/{args.name}")
    print()

    results = model.train(**train_kwargs)

    # ── Copy best weights to models/ ─────────────────────────────────────────
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        dest = Path("models") / "court_segmentation" / "best.pt"
        os.makedirs(dest.parent, exist_ok=True)
        shutil.copy2(best_weights, dest)
        print(f"\nBest weights copied to {dest}")
    else:
        print(f"\n⚠️  best.pt not found at {best_weights} — skipping copy to models/")

    # ── Final validation pass ────────────────────────────────────────────────
    print("\nRunning final validation on best model...")
    best_model = YOLO(str(best_weights)) if best_weights.exists() else model
    best_model.val(data=str(data_path))

    print("\n=== Training Complete ===")
    print(f"Best weights:     {best_weights}")
    print(f"Canonical copy:   models/court_segmentation/best.pt")
    print(f"All results:      {args.project}/{args.name}/")

    return results


if __name__ == "__main__":
    args = get_args()
    train(args)
