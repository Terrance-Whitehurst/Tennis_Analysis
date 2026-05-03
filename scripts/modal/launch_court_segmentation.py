"""
Modal launcher for tennis court instance segmentation training.

Runs YOLO11-seg training on a remote GPU via Modal serverless infrastructure.
The 24 MB dataset is embedded directly into the container image layer at build
time (add_local_dir), so there is no separate upload step and no cold-start
volume-mount latency.

Prerequisites:
    pip install modal
    modal token new          # one-time auth — opens browser

Usage:
    # Default: A10G GPU, 100 epochs, foreground (streaming output)
    python scripts/modal/launch_court_segmentation.py

    # Larger GPU, more epochs
    python scripts/modal/launch_court_segmentation.py --gpu A100 --epochs 150

    # Background / detached mode — returns immediately
    python scripts/modal/launch_court_segmentation.py --detach

    # Different model size
    python scripts/modal/launch_court_segmentation.py --model yolo11l-seg.pt

    # Custom run name (used as the subfolder in the output volume)
    python scripts/modal/launch_court_segmentation.py --name v2_large

After training, retrieve weights with:
    modal volume get tennis-court-seg-outputs run/best.pt models/court_segmentation/best.pt
    modal volume get tennis-court-seg-outputs run/last.pt  models/court_segmentation/last.pt

(The exact commands are also printed at the end of every run.)

─────────────────────────────────────────────────────────────────────────────
DESIGN DECISIONS
─────────────────────────────────────────────────────────────────────────────
GPU: A10G (default)
    24 GB VRAM. Auto-batch at imgsz=640 should land ~16–24 images/batch.
    100 epochs ≈ 30–45 min ≈ $0.55–$0.70. Override with --gpu A100 for
    larger experiments or --gpu T4/L4 for budget runs.

Dataset bake-in: image.add_local_dir (not a separate Volume)
    The processed dataset is 24 MB. Embedding it in the container image layer
    is the right call — zero upload latency, no Volume mount on cold start,
    and image layers are cached across runs so you only pay the bake cost once.

Output Volume: tennis-court-seg-outputs
    Persists best.pt and last.pt between runs. Survives container shutdown.
    Retrieve locally with `modal volume get`.

Dynamic GPU: parsed from --gpu before module-level decorator evaluation
    Modal's @app.function decorator is evaluated at import time, so the GPU
    type must be known before Python reaches the decorator. We parse --gpu
    from sys.argv early (before argparse) and use the result in the decorator.
    This is the standard pattern for user-configurable GPU types in Modal scripts.

Augmentations: identical to src/training/train_court_segmentation.py
    See that file for the full rationale comment block.
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
    """Return the --gpu value from sys.argv, or the default 'A10G'."""
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token == "--gpu" and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith("--gpu="):
            return token.split("=", 1)[1]
    return "A10G"


GPU_TYPE: str = _gpu_from_argv()

# Valid GPU choices — used for CLI validation and the reminder comment below.
_VALID_GPUS = ("T4", "L4", "A10G", "A100", "H100")

# ── Modal primitives ──────────────────────────────────────────────────────────

app = modal.App("tennis-court-segmentation")

# Container image: slim Debian + ML stack + baked-in dataset.
# Image build is deferred until the first .remote()/.spawn() call, so the
# local converter runs (see ensure_dataset_converted()) before Modal reads
# data/processed/court_segmentation_yolo/.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Ultralytics declares opencv-python (non-headless) as a hard dep, which
    # links against libGL/libglib at import time. debian_slim ships neither,
    # so we install the system libs rather than fight the transitive resolve.
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.3",
        "torch>=2.2",
        "torchvision",
        "pyyaml",
        "numpy",
        "pillow",
        "opencv-python-headless",
    )
    # Bake the 24 MB processed dataset into the image layer.
    # Run `python scripts/convert_coco_to_yolo_seg.py` first if this
    # directory does not yet exist (the __main__ block does this automatically).
    .add_local_dir(
        "data/processed/court_segmentation_yolo",
        remote_path="/data/court_segmentation_yolo",
    )
)

# Persistent output volume — survives container shutdown.
output_vol = modal.Volume.from_name(
    "tennis-court-seg-outputs",
    create_if_missing=True,
)

_VOLUME_MOUNT = "/outputs"


# ── Remote training function ──────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,          # set at import time from --gpu arg (default A10G)
    timeout=7_200,         # 2 hours; 100 epochs on A10G ≈ 35 min in practice
    volumes={_VOLUME_MOUNT: output_vol},
)
def train_remote(
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    name: str,
) -> dict:
    """
    Run YOLO11-seg training inside the Modal container.

    The dataset lives at /data/court_segmentation_yolo (baked into the image).
    Weights are written to /outputs/<name>/ and committed to the Modal volume.
    """
    import shutil
    import tempfile
    from pathlib import Path as _Path

    import yaml
    from ultralytics import YOLO

    # ── Build a container-local data.yaml ────────────────────────────────────
    # We cannot use the host configs/court_segmentation.yaml directly because
    # its `path` points to a relative local path.  Write a fresh one here that
    # uses the absolute container path /data/court_segmentation_yolo.
    data_cfg = {
        "path": "/data/court_segmentation_yolo",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 3,
        "names": {0: "doubles_alley", 1: "no_mans_land", 2: "service_box"},
    }
    cfg_file = _Path(tempfile.mkdtemp()) / "court_segmentation.yaml"
    with open(cfg_file, "w") as fh:
        yaml.dump(data_cfg, fh)

    # ── Warn about the tiny validation split ─────────────────────────────────
    val_dir = _Path("/data/court_segmentation_yolo/valid/images")
    if val_dir.exists():
        n_val = len(list(val_dir.glob("*")))
        if n_val < 10:
            print("\n" + "=" * 60)
            print(f"⚠️  WARNING: val split has only {n_val} image(s).")
            print("   mAP scores during training will be extremely noisy.")
            print("   Training proceeds normally — ignore the val metrics.")
            print("=" * 60 + "\n")

    # ── Training banner ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Tennis Court Instance Segmentation — Modal Training")
    print(f"  Model:   {model}")
    print(f"  Epochs:  {epochs}  |  imgsz: {imgsz}  |  batch: {batch} (auto=-1)")
    print(f"  Run:     {name}")
    print("=" * 60 + "\n")

    # ── Load model and run training ───────────────────────────────────────────
    yolo = YOLO(model)
    results = yolo.train(
        data=str(cfg_file),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=25,
        optimizer="auto",       # AdamW for YOLO11
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
        # Flip: never vertical (gravity), always horizontal (symmetric court)
        flipud=0.0,
        fliplr=0.5,
        # Composition: critical with only 427 training images
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        close_mosaic=10,
        # ── Output ───────────────────────────────────────────────────────────
        project="/tmp/runs",
        name=name,
        exist_ok=True,
        verbose=True,
        plots=True,
    )

    # ── Copy weights to the persistent volume ────────────────────────────────
    weights_dir = _Path("/tmp/runs") / name / "weights"
    out_dir = _Path(_VOLUME_MOUNT) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for fname in ("best.pt", "last.pt"):
        src = weights_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            saved.append(fname)
            print(f"  Saved {fname} → {out_dir / fname}")
        else:
            print(f"  ⚠️  {fname} not found — skipping")

    # Commit is required to make new/updated files visible outside the container
    output_vol.commit()

    print("\n" + "=" * 60)
    print("✅  Training complete.")
    print(f"    Volume : tennis-court-seg-outputs")
    print(f"    Path   : {name}/")
    print(f"    Files  : {', '.join(saved) if saved else '(none copied)'}")
    print("=" * 60 + "\n")

    # Return a lightweight summary (Ultralytics results_dict if available)
    metrics: dict = {}
    if hasattr(results, "results_dict"):
        metrics = dict(results.results_dict)

    return {"run_name": name, "weights_saved": saved, "metrics": metrics}


# ── Local helpers ─────────────────────────────────────────────────────────────

def ensure_dataset_converted() -> None:
    """
    Check that data/processed/court_segmentation_yolo/ exists.
    If not, call the COCO-to-YOLO-seg converter automatically.

    This runs *before* Modal builds the container image, so add_local_dir
    finds the files when it needs them.
    """
    processed = Path("data/processed/court_segmentation_yolo/train/images")
    if processed.exists():
        return

    print("Processed dataset not found at data/processed/court_segmentation_yolo/")
    print("Running COCO-to-YOLO-seg converter automatically...\n")

    # Ensure project root is on the path for the import
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Preserve the real argv; converter uses argparse with its own defaults
    saved_argv = sys.argv[:]
    sys.argv = ["convert_coco_to_yolo_seg.py"]
    try:
        from scripts.convert_coco_to_yolo_seg import main as _convert
        _convert()
    finally:
        sys.argv = saved_argv

    if not processed.exists():
        raise FileNotFoundError(
            "\nConversion failed — processed dataset still missing.\n"
            "Run manually:  python scripts/convert_coco_to_yolo_seg.py\n"
            "Then retry:    python scripts/modal/launch_court_segmentation.py"
        )

    print("\nConversion complete.\n")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch tennis court segmentation training on Modal",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="yolo11m-seg.pt",
        help=(
            "Pretrained YOLO seg checkpoint to fine-tune. "
            "Options: yolo11{n,s,m,l,x}-seg.pt"
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (source images are 854×480)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 = auto-batch based on VRAM)",
    )
    parser.add_argument(
        "--gpu",
        default="A10G",
        choices=list(_VALID_GPUS),
        help=(
            "Modal GPU type. NOTE: this is parsed early from sys.argv to set "
            "the @app.function decorator — changing it here changes the actual GPU."
        ),
    )
    parser.add_argument(
        "--name",
        default="run",
        help="Run name — used as the subfolder inside the Modal output volume",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Spawn the job in background and return immediately. "
            "Track progress at https://modal.com/apps/tennis-court-segmentation"
        ),
    )
    return parser.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = get_args()

    # Step 1 — Ensure the processed dataset exists locally.
    # Modal's add_local_dir reads from disk when it builds the image, which
    # happens just before the first .remote() / .spawn() call below.
    ensure_dataset_converted()

    # Step 2 — Launch banner
    print("\n" + "=" * 60)
    print("Tennis Court Segmentation — Modal Launch")
    print(f"  GPU    : {GPU_TYPE}")
    print(f"  Model  : {args.model}")
    print(f"  Epochs : {args.epochs}  |  imgsz: {args.imgsz}  |  batch: {args.batch}")
    print(f"  Name   : {args.name}")
    print(f"  Mode   : {'detached (background)' if args.detach else 'foreground (streaming)'}")
    print("=" * 60 + "\n")

    # Step 3 — Run on Modal
    with modal.enable_output():
        with app.run():
            if args.detach:
                # Spawn in background — returns a FunctionCall handle immediately
                call = train_remote.spawn(
                    model=args.model,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    name=args.name,
                )
                print("Job spawned in background.")
                print(
                    "Track progress at: "
                    "https://modal.com/apps/tennis-court-segmentation"
                )
            else:
                # Foreground — blocks until training finishes, streams logs
                result = train_remote.remote(
                    model=args.model,
                    epochs=args.epochs,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    name=args.name,
                )
                if result.get("metrics"):
                    print("\nFinal metrics:")
                    for k, v in result["metrics"].items():
                        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Step 4 — Print retrieval commands (always shown, detach or foreground)
    print("\n" + "=" * 60)
    print("Retrieve weights locally:")
    print(
        f"  modal volume get tennis-court-seg-outputs "
        f"{args.name}/best.pt models/court_segmentation/best.pt"
    )
    print(
        f"  modal volume get tennis-court-seg-outputs "
        f"{args.name}/last.pt models/court_segmentation/last.pt"
    )
    print()
    print("List all files in the volume:")
    print("  modal volume ls tennis-court-seg-outputs")
    print("=" * 60 + "\n")
