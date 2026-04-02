"""
Convert COCO keypoint annotations to YOLO pose format.

COCO keypoint format (per annotation in JSON):
    "bbox": [x, y, w, h]  (pixels)
    "keypoints": [x1, y1, v1, x2, y2, v2, ..., xN, yN, vN]  (pixels, visibility)

YOLO pose format (per line in .txt file):
    class_id cx cy w h x1 y1 v1 x2 y2 v2 ... xN yN vN
    All coordinates normalized to [0, 1] relative to image dimensions.
    Visibility: 0=not labeled, 1=labeled but occluded, 2=labeled and visible

This script converts the Tennis_Court_Keypoint dataset:
    - 14 keypoints per court annotation
    - Categories: tennis_court (id=0, no keypoints), tenniscourt (id=1, with keypoints)
    - We only convert category_id=1 annotations (the ones with keypoints)

Output structure:
    data/processed/court_keypoint_yolo/
    ├── train/
    │   ├── images/  (symlinks to raw images)
    │   └── labels/  (converted .txt files)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Usage:
    python scripts/convert_coco_to_yolo_kpt.py
    python scripts/convert_coco_to_yolo_kpt.py --input data/raw/Tennis_Court_Keypoint --output data/processed/court_keypoint_yolo
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def convert_coco_to_yolo_kpt(coco_json_path, images_dir, output_images_dir, output_labels_dir, num_keypoints=14):
    """Convert a single COCO annotation file to YOLO keypoint format.

    Args:
        coco_json_path: Path to _annotations.coco.json
        images_dir: Directory containing the source images
        output_images_dir: Directory to copy/symlink images to
        output_labels_dir: Directory to write YOLO .txt label files
        num_keypoints: Number of keypoints per annotation
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image lookup: id -> {file_name, width, height}
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
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

        # Copy image to output directory
        src_img = Path(images_dir) / file_name
        dst_img = Path(output_images_dir) / file_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Convert annotations for this image
        anns = annotations_by_image.get(img_id, [])
        label_name = Path(file_name).stem + ".txt"
        label_path = Path(output_labels_dir) / label_name

        lines = []
        for ann in anns:
            # Only convert annotations that have keypoints (category_id=1)
            if "keypoints" not in ann or len(ann["keypoints"]) == 0:
                skipped_count += 1
                continue

            keypoints = ann["keypoints"]
            if len(keypoints) != num_keypoints * 3:
                skipped_count += 1
                continue

            # Convert bbox from COCO [x, y, w, h] to YOLO [cx, cy, w, h] normalized
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / img_w
            cy = (by + bh / 2) / img_h
            nw = bw / img_w
            nh = bh / img_h

            # Clamp bbox to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            # Use class 0 for all court keypoint annotations in YOLO
            # (single class detection)
            class_id = 0

            # Convert keypoints: normalize x, y to [0, 1], keep visibility as-is
            kpt_parts = []
            for k in range(num_keypoints):
                kx = keypoints[k * 3] / img_w
                ky = keypoints[k * 3 + 1] / img_h
                kv = int(keypoints[k * 3 + 2])

                # Clamp coordinates
                kx = max(0.0, min(1.0, kx))
                ky = max(0.0, min(1.0, ky))

                # If visibility is 0, zero out coordinates
                if kv == 0:
                    kx, ky = 0.0, 0.0

                kpt_parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])

            line = f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kpt_parts)
            lines.append(line)
            converted_count += 1

        # Write label file (even if empty — YOLO expects a file per image)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    return converted_count, skipped_count


def main():
    parser = argparse.ArgumentParser(description="Convert COCO keypoint annotations to YOLO pose format")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/Tennis_Court_Keypoint",
        help="Path to COCO keypoint dataset root",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/court_keypoint_yolo",
        help="Output directory for YOLO format dataset",
    )
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=14,
        help="Number of keypoints per annotation",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    print(f"Converting COCO keypoint annotations to YOLO format")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Keypoints: {args.num_keypoints}")
    print()

    for split in ["train", "valid", "test"]:
        coco_json = input_dir / split / "_annotations.coco.json"
        if not coco_json.exists():
            print(f"  [{split}] Skipped — no annotations found at {coco_json}")
            continue

        images_dir = input_dir / split
        out_images = output_dir / split / "images"
        out_labels = output_dir / split / "labels"

        converted, skipped = convert_coco_to_yolo_kpt(
            coco_json, images_dir, out_images, out_labels, args.num_keypoints
        )
        print(f"  [{split}] Converted: {converted} annotations, Skipped: {skipped}")

    print(f"\nDone! YOLO dataset written to {output_dir}")
    print(f"Use configs/court_keypoint.yaml for training with Ultralytics.")


if __name__ == "__main__":
    main()
