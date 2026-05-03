"""
Modal launcher for tennis court instance segmentation with RF-DETR-Seg.

Fine-tunes Roboflow's RF-DETR-Seg (DINOv2 backbone, transformer head) on a
remote GPU via Modal serverless infrastructure. The Roboflow-export COCO
dataset is embedded directly into the container image layer at build time
(add_local_dir), so there is no separate upload step.

Prerequisites:
    pip install modal
    modal token new          # one-time auth — opens browser

Usage:
    # Default: RFDETRSegMedium, A10G GPU, 100 epochs, foreground (streaming)
    python scripts/modal/launch_court_segmentation_rfdetr.py

    # Larger model + bigger GPU
    python scripts/modal/launch_court_segmentation_rfdetr.py --model large --gpu A100

    # Background / detached mode
    python scripts/modal/launch_court_segmentation_rfdetr.py --detach

    # Custom run name (used as output sub-folder in the Modal volume)
    python scripts/modal/launch_court_segmentation_rfdetr.py --name v2

After training, retrieve weights with:
    modal volume get tennis-court-seg-rfdetr-outputs run/checkpoint_best_ema.pth \
        models/court_segmentation/rfdetr_best.pth

(The exact retrieval commands are also printed at the end of every run.)

─────────────────────────────────────────────────────────────────────────────
DESIGN NOTES
─────────────────────────────────────────────────────────────────────────────
Model: RFDETRSegMedium (default)
    35.7M params @ 432x432. Same parameter budget as YOLO11m-seg, but RF-DETR-Seg
    reports ~68.4 COCO mAP50 vs YOLO11m-seg's ~60.0 — meaningful headroom.
    Override with --model {nano,small,medium,large,xlarge,2xlarge} as needed.

Dataset: data/raw/court_segmentation (Roboflow COCO export, baked into image)
    RF-DETR auto-detects COCO format from the directory layout. No conversion
    step needed — the train/valid/test/_annotations.coco.json files ship as-is.

GPU: A10G default (~$0.36/hr)
    RF-DETR-Seg-M trains comfortably at batch=4, grad_accum=4 on 22 GB. Bump
    to A100 with --gpu A100 if you go to Large/XLarge or want to widen batch.

Output Volume: tennis-court-seg-rfdetr-outputs
    Separate from the YOLO volume so checkpoints don't collide.

Effective batch: batch_size × grad_accum_steps = 4 × 4 = 16 (Roboflow default)
    Don't change without reason — RF-DETR's defaults are tuned for this budget.

Augmentations: leave RF-DETR defaults
    The library handles transformer-friendly augmentation internally; manual
    overrides risk degrading results. See learn/train/augmentations/ if needed.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

# ── Parse --gpu early — Modal decorators are evaluated at import time ─────────
# The decorator must know the GPU type before Python reaches it, so we scan
# sys.argv directly rather than waiting for argparse.

def _gpu_from_argv() -> str:
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == "--gpu" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--gpu="):
            return token.split("=", 1)[1]
    return "A10G"


GPU_TYPE: str = _gpu_from_argv()
_VALID_GPUS = ("T4", "L4", "A10G", "A100", "H100")

# ── Model size → RF-DETR class name ──────────────────────────────────────────
_MODEL_CLASSES = {
    "nano":    "RFDETRSegNano",
    "small":   "RFDETRSegSmall",
    "medium":  "RFDETRSegMedium",
    "large":   "RFDETRSegLarge",
    "xlarge":  "RFDETRSegXLarge",
    "2xlarge": "RFDETRSeg2XLarge",
}

# ── Modal primitives ──────────────────────────────────────────────────────────

app = modal.App("tennis-court-segmentation-rfdetr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # rfdetr → torchvision → opencv-python (non-headless) needs libGL/libglib.
    .apt_install("libgl1", "libglib2.0-0")
    # rfdetr[train,loggers] pulls in pytorch_lightning + TB/W&B; required for
    # .train() — the bare `rfdetr` install only ships inference dependencies.
    .pip_install(
        "rfdetr[train,loggers]>=1.6",
        "supervision>=0.25",
        "torch>=2.2",
        "torchvision",
    )
    # Bake the Roboflow COCO export (~30 MB of JPG + tiny annotation JSONs).
    .add_local_dir(
        "data/raw/court_segmentation",
        remote_path="/data/court_segmentation",
    )
)

output_vol = modal.Volume.from_name(
    "tennis-court-seg-rfdetr-outputs",
    create_if_missing=True,
)

_VOLUME_MOUNT = "/outputs"


# ── Remote training function ──────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=10_800,         # 3 hours; RF-DETR-Seg-M @ 100 epochs ≈ 60-90 min on A10G
    volumes={_VOLUME_MOUNT: output_vol},
)
def train_remote(
    model_size: str,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    name: str,
    early_stopping: bool,
) -> dict:
    """Run RF-DETR-Seg fine-tuning inside the Modal container."""
    import importlib
    import shutil
    from pathlib import Path as _Path

    # ── Resolve model class dynamically from --model size flag ───────────────
    rfdetr_module = importlib.import_module("rfdetr")
    cls_name = _MODEL_CLASSES[model_size]
    if not hasattr(rfdetr_module, cls_name):
        raise RuntimeError(
            f"rfdetr does not expose {cls_name}. Upgrade rfdetr (>=1.6) or pick "
            f"a different --model. Available: {list(_MODEL_CLASSES)}"
        )
    ModelCls = getattr(rfdetr_module, cls_name)

    # ── Warn about the tiny validation split ─────────────────────────────────
    val_dir = _Path("/data/court_segmentation/valid")
    n_val = len(list(val_dir.glob("*.jpg"))) if val_dir.exists() else 0
    if n_val < 10:
        print("\n" + "=" * 60)
        print(f"⚠️  WARNING: val split has only {n_val} image(s).")
        print("   Eval mAP curves will be extremely noisy.")
        print("   Training proceeds — calibrate expectations on val numbers.")
        print("=" * 60 + "\n")

    print("\n" + "=" * 60)
    print("Tennis Court Segmentation — RF-DETR-Seg Modal Training")
    print(f"  Model    : {cls_name}")
    print(f"  Epochs   : {epochs}")
    print(f"  Batch    : {batch_size} × grad_accum={grad_accum_steps} "
          f"(effective {batch_size * grad_accum_steps})")
    print(f"  LR       : {lr}")
    print(f"  Run      : {name}")
    print(f"  EarlyStop: {early_stopping}")
    print("=" * 60 + "\n")

    output_dir = _Path("/tmp/runs") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ModelCls()
    model.train(
        dataset_dir="/data/court_segmentation",
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
        output_dir=str(output_dir),
        tensorboard=True,
        early_stopping=early_stopping,
    )

    # ── Copy every checkpoint to the persistent volume ───────────────────────
    out_dir = _Path(_VOLUME_MOUNT) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for ckpt in output_dir.rglob("*.pth"):
        dest = out_dir / ckpt.relative_to(output_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ckpt, dest)
        saved.append(str(ckpt.relative_to(output_dir)))
        print(f"  Saved {ckpt.name} → {dest}")

    # Also copy any tensorboard logs / metrics.json so plots survive.
    for log_pattern in ("*.json", "events.out.tfevents.*"):
        for f in output_dir.rglob(log_pattern):
            dest = out_dir / f.relative_to(output_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)

    output_vol.commit()

    print("\n" + "=" * 60)
    print("✅  Training complete.")
    print(f"    Volume : tennis-court-seg-rfdetr-outputs")
    print(f"    Path   : {name}/")
    print(f"    Files  : {len(saved)} checkpoint(s)")
    print("=" * 60 + "\n")

    return {"run_name": name, "model": cls_name, "checkpoints_saved": saved}


# ── CLI ──────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch tennis court RF-DETR-Seg training on Modal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=list(_MODEL_CLASSES),
        help="RF-DETR-Seg size — picks the class (RFDETRSeg{Nano..2XLarge})",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--gpu",
        default="A10G",
        choices=list(_VALID_GPUS),
        help=(
            "Modal GPU type. NOTE: parsed early from sys.argv to set the "
            "@app.function decorator at import time."
        ),
    )
    parser.add_argument("--name", default="run", help="Run name / output sub-folder")
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping (defaults to on)",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Spawn job in background and return immediately",
    )
    return parser.parse_args()


def ensure_dataset_present() -> None:
    """Verify the raw COCO dataset is on disk before Modal builds the image."""
    raw = Path("data/raw/court_segmentation/train/_annotations.coco.json")
    if not raw.exists():
        raise FileNotFoundError(
            f"\nExpected Roboflow COCO export at data/raw/court_segmentation/.\n"
            f"Missing: {raw}\n"
            f"Confirm the dataset is on disk before launching."
        )


if __name__ == "__main__":
    args = get_args()
    ensure_dataset_present()

    print("\n" + "=" * 60)
    print("Tennis Court RF-DETR-Seg — Modal Launch")
    print(f"  GPU      : {GPU_TYPE}")
    print(f"  Model    : {_MODEL_CLASSES[args.model]} (--model {args.model})")
    print(f"  Epochs   : {args.epochs}  |  batch: {args.batch_size}  "
          f"|  grad_accum: {args.grad_accum_steps}  |  lr: {args.lr}")
    print(f"  Name     : {args.name}")
    print(f"  Mode     : {'detached (background)' if args.detach else 'foreground (streaming)'}")
    print("=" * 60 + "\n")

    with modal.enable_output():
        with app.run():
            train_kwargs = dict(
                model_size=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum_steps,
                lr=args.lr,
                name=args.name,
                early_stopping=not args.no_early_stopping,
            )
            if args.detach:
                call = train_remote.spawn(**train_kwargs)
                print("Job spawned in background.")
                print(
                    "Track progress at: "
                    "https://modal.com/apps/tennis-court-segmentation-rfdetr"
                )
            else:
                result = train_remote.remote(**train_kwargs)
                print(f"\nRun summary: {result}")

    print("\n" + "=" * 60)
    print("Retrieve weights locally:")
    print(
        f"  modal volume get tennis-court-seg-rfdetr-outputs "
        f"{args.name}/checkpoint_best_ema.pth "
        f"models/court_segmentation/rfdetr_best.pth"
    )
    print()
    print("List all files in the volume:")
    print("  modal volume ls tennis-court-seg-rfdetr-outputs")
    print("=" * 60 + "\n")
