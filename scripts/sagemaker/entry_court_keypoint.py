"""
SageMaker entry point for YOLO-Pose court keypoint detection training.

SageMaker invokes this script inside a PyTorch DLC container on a GPU instance.
It handles:
    1. Installing ultralytics at runtime
    2. Converting COCO keypoint annotations to YOLO format
    3. Generating the dataset YAML pointing to the converted data
    4. Running YOLO-Pose training
    5. Copying model artifacts to /opt/ml/model/ for S3 upload

Data channels:
    /opt/ml/input/data/training/    <- S3 dataset (COCO keypoint format)
        ├── train/
        │   ├── _annotations.coco.json
        │   └── *.jpg
        ├── valid/
        │   ├── _annotations.coco.json
        │   └── *.jpg
        └── test/
            ├── _annotations.coco.json
            └── *.jpg

    /opt/ml/model/                  <- Trained model artifacts -> S3

Usage (launched by SageMaker, not directly):
    python entry_court_keypoint.py --epochs 100 --batch 16 --imgsz 640
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def install_dependencies():
    """Install ultralytics inside the SageMaker container at runtime."""
    print("Installing ultralytics...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "ultralytics"
    ])
    print("ultralytics installed successfully.")


def convert_coco_to_yolo_kpt(coco_json_path, images_dir, output_images_dir, output_labels_dir, num_keypoints=14):
    """Convert COCO keypoint annotations to YOLO pose format.

    Inline version of scripts/convert_coco_to_yolo_kpt.py so the entry point
    is self-contained (no dependency on the project source tree).
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    converted_count = 0
    skipped_count = 0

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img = Path(images_dir) / file_name
        dst_img = Path(output_images_dir) / file_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        anns = annotations_by_image.get(img_id, [])
        label_name = Path(file_name).stem + ".txt"
        label_path = Path(output_labels_dir) / label_name

        lines = []
        for ann in anns:
            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                skipped_count += 1
                continue

            keypoints = ann["keypoints"]
            if len(keypoints) != num_keypoints * 3:
                skipped_count += 1
                continue

            bx, by, bw, bh = ann["bbox"]
            cx = max(0.0, min(1.0, (bx + bw / 2) / img_w))
            cy = max(0.0, min(1.0, (by + bh / 2) / img_h))
            nw = max(0.0, min(1.0, bw / img_w))
            nh = max(0.0, min(1.0, bh / img_h))

            class_id = 0
            kpt_parts = []
            for k in range(num_keypoints):
                kx = keypoints[k * 3] / img_w
                ky = keypoints[k * 3 + 1] / img_h
                kv = int(keypoints[k * 3 + 2])
                kx = max(0.0, min(1.0, kx))
                ky = max(0.0, min(1.0, ky))
                if kv == 0:
                    kx, ky = 0.0, 0.0
                kpt_parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])

            line = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kpt_parts)
            lines.append(line)
            converted_count += 1

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    return converted_count, skipped_count


def prepare_yolo_dataset(coco_data_dir, yolo_output_dir):
    """Convert COCO keypoint dataset to YOLO format and write dataset YAML."""
    coco_data_dir = Path(coco_data_dir)
    yolo_output_dir = Path(yolo_output_dir)

    print("Converting COCO keypoint annotations to YOLO format...")
    for split in ["train", "valid", "test"]:
        coco_json = coco_data_dir / split / "_annotations.coco.json"
        if not coco_json.exists():
            print(f"  [{split}] Skipped — no annotations")
            continue

        out_images = yolo_output_dir / split / "images"
        out_labels = yolo_output_dir / split / "labels"

        converted, skipped = convert_coco_to_yolo_kpt(
            str(coco_json), str(coco_data_dir / split),
            str(out_images), str(out_labels), num_keypoints=14
        )
        print(f"  [{split}] Converted: {converted}, Skipped: {skipped}")

    # Write dataset YAML
    yaml_path = yolo_output_dir / "dataset.yaml"
    yaml_content = f"""# Auto-generated YOLO dataset config for SageMaker
path: {yolo_output_dir}
train: train/images
val: valid/images
test: test/images

kpt_shape: [14, 3]

names:
  0: tenniscourt

skeleton:
  - [0, 4]
  - [4, 6]
  - [6, 1]
  - [1, 3]
  - [3, 7]
  - [7, 5]
  - [5, 2]
  - [2, 0]
  - [4, 8]
  - [8, 10]
  - [10, 5]
  - [6, 9]
  - [9, 11]
  - [11, 7]
  - [8, 12]
  - [12, 9]
  - [10, 13]
  - [13, 11]
  - [12, 13]
"""
    yaml_path.write_text(yaml_content)
    print(f"Dataset YAML written to {yaml_path}")

    return str(yaml_path)


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker environment paths
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--num_gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", 1)))

    # Model
    parser.add_argument("--model", type=str, default="yolo11m-pose.pt")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--optimizer", type=str, default="auto")

    # Augmentation
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)

    # Loss weights
    parser.add_argument("--pose", type=float, default=12.0)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=0.5)
    parser.add_argument("--dfl", type=float, default=1.5)

    parser.add_argument("--workers", type=int, default=8)

    return parser.parse_args()


def main():
    args = parse_args()

    # Install ultralytics (not in base PyTorch DLC)
    install_dependencies()

    from ultralytics import YOLO

    # Validate data channel
    data_dir = args.data_dir
    print(f"Data directory: {data_dir}")
    print(f"Contents: {os.listdir(data_dir)}")

    # Convert COCO -> YOLO format inside the container
    yolo_data_dir = "/opt/ml/input/data/yolo_converted"
    dataset_yaml = prepare_yolo_dataset(data_dir, yolo_data_dir)

    # Training output directory (inside container)
    project_dir = "/opt/ml/output/data"
    run_name = "court_keypoint"

    print(f"\n{'='*60}")
    print(f"YOLO-Pose Court Keypoint Training")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch:       {args.batch}")
    print(f"Image size:  {args.imgsz}")
    print(f"LR:          {args.lr0} -> {args.lr0 * args.lrf}")
    print(f"Pose weight: {args.pose}")
    print(f"GPUs:        {args.num_gpus}")
    print(f"{'='*60}\n")

    # Initialize and train
    model = YOLO(args.model)

    train_kwargs = dict(
        data=dataset_yaml,
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
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        scale=args.scale,
        pose=args.pose,
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        workers=args.workers,
        device="0",
        verbose=True,
        plots=True,
    )

    results = model.train(**train_kwargs)

    # Copy model artifacts to SM_MODEL_DIR for S3 upload
    print(f"\nCopying model artifacts to {args.model_dir}...")
    os.makedirs(args.model_dir, exist_ok=True)

    weights_dir = Path(project_dir) / run_name / "weights"
    if weights_dir.exists():
        for pt_file in weights_dir.glob("*.pt"):
            shutil.copy2(pt_file, os.path.join(args.model_dir, pt_file.name))
            print(f"  Copied {pt_file.name}")

        # Copy best as the canonical model.pt
        best = weights_dir / "best.pt"
        if best.exists():
            shutil.copy2(best, os.path.join(args.model_dir, "model.pt"))
            print("  Best model saved as model.pt")

    # Copy training results (plots, metrics CSV)
    results_dir = Path(project_dir) / run_name
    for f in results_dir.glob("*.csv"):
        shutil.copy2(f, os.path.join(args.model_dir, f.name))
    for f in results_dir.glob("*.png"):
        shutil.copy2(f, os.path.join(args.model_dir, f.name))

    print("\nTraining complete! Model artifacts will be uploaded to S3.")


if __name__ == "__main__":
    main()
