"""
Convert COCO segmentation annotations to YOLO instance-segmentation format.

COCO segmentation format (per annotation in JSON):
    "segmentation": [[x1, y1, x2, y2, ..., xN, yN]]  (absolute pixels, single polygon)
    "bbox": [x, y, w, h]  (absolute pixels, used only as fallback — not needed for seg)

YOLO seg format (per line in .txt file):
    class_id x1 y1 x2 y2 ... xN yN
    All coordinates normalized to [0, 1] relative to image dimensions.
    No bounding box on the line — Ultralytics derives it from the polygon.

Dataset: Tennis Court Segmentation (Roboflow, 2026 Acapulco ATP)
    Source: data/raw/court_segmentation/
    3 active classes (COCO id=0 "courts" is an unused hierarchical parent):
        COCO id 1  doubles_alley  → YOLO class 0
        COCO id 2  no_mans_land   → YOLO class 1
        COCO id 3  service_box    → YOLO class 2

Output structure:
    data/processed/court_segmentation_yolo/
    ├── train/
    │   ├── images/   (copied from raw)
    │   └── labels/   (converted .txt files)
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

Usage:
    python scripts/convert_coco_to_yolo_seg.py
    python scripts/convert_coco_to_yolo_seg.py --input data/raw/court_segmentation --output data/processed/court_segmentation_yolo
    python scripts/convert_coco_to_yolo_seg.py --force   # overwrite existing output
"""

import argparse
import json
import os
import shutil
from pathlib import Path

# COCO category id → YOLO class id mapping.
# id=0 "courts" is a zero-annotation parent class — intentionally excluded.
CATEGORY_REMAP = {
    1: 0,  # doubles_alley
    2: 1,  # no_mans_land
    3: 2,  # service_box
}

CLASS_NAMES = {
    0: "doubles_alley",
    1: "no_mans_land",
    2: "service_box",
}


def convert_split(
    coco_json_path: Path,
    images_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
) -> tuple[int, int, int]:
    """Convert one split's COCO segmentation JSON to YOLO-seg .txt files.

    Returns:
        (num_images, num_annotations_converted, num_annotations_skipped)
    """
    with open(coco_json_path) as f:
        coco = json.load(f)

    # Build image lookup: id → {file_name, width, height}
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image id
    anns_by_image: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    converted = 0
    skipped = 0

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Copy image to output structure (copy not symlink — safer for Modal mounts)
        src_img = images_dir / file_name
        dst_img = output_images_dir / file_name
        if src_img.exists() and not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Build YOLO label lines for this image
        anns = anns_by_image.get(img_id, [])
        lines: list[str] = []

        for ann in anns:
            coco_cat_id = ann.get("category_id")

            # Skip unmapped categories (e.g., the "courts" parent class, id=0)
            if coco_cat_id not in CATEGORY_REMAP:
                skipped += 1
                continue

            yolo_class_id = CATEGORY_REMAP[coco_cat_id]

            # COCO segmentation is a list of polygons per annotation.
            # Roboflow exports typically produce exactly one polygon per annotation.
            # If there are multiple, we emit one YOLO line per polygon (Ultralytics supports this).
            segmentation = ann.get("segmentation", [])
            if not segmentation:
                skipped += 1
                continue

            for polygon in segmentation:
                # polygon is a flat list: [x1, y1, x2, y2, ..., xN, yN]
                if len(polygon) < 6:  # need at least 3 points
                    skipped += 1
                    continue

                # Normalize all coordinates to [0, 1]
                norm_coords: list[str] = []
                for i in range(0, len(polygon), 2):
                    px = polygon[i]
                    py = polygon[i + 1]
                    nx = max(0.0, min(1.0, px / img_w))
                    ny = max(0.0, min(1.0, py / img_h))
                    norm_coords.append(f"{nx:.6f}")
                    norm_coords.append(f"{ny:.6f}")

                line = f"{yolo_class_id} " + " ".join(norm_coords)
                lines.append(line)
                converted += 1

        # Write label file (empty file if no annotations — YOLO expects a file per image)
        label_stem = Path(file_name).stem
        label_path = output_labels_dir / f"{label_stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    return len(images), converted, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO segmentation annotations to YOLO instance-seg format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/court_segmentation",
        help="Path to COCO segmentation dataset root (with train/valid/test sub-dirs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/court_segmentation_yolo",
        help="Output directory for YOLO-seg format dataset",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion even if output directory already exists",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Idempotency check: skip if already converted (unless --force)
    if output_dir.exists() and not args.force:
        print(f"Output already exists at {output_dir}")
        print("  Use --force to re-convert.")
        return

    if args.force and output_dir.exists():
        print(f"--force specified, removing existing output at {output_dir}")
        shutil.rmtree(output_dir)

    print("Converting COCO segmentation annotations to YOLO-seg format")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print()
    print("  Class mapping:")
    for coco_id, yolo_id in CATEGORY_REMAP.items():
        print(f"    COCO id {coco_id} → YOLO id {yolo_id}  ({CLASS_NAMES[yolo_id]})")
    print()

    total_images = 0
    total_converted = 0
    total_skipped = 0

    for split in ["train", "valid", "test"]:
        coco_json = input_dir / split / "_annotations.coco.json"
        if not coco_json.exists():
            print(f"  [{split:5s}] Skipped — no annotation file at {coco_json}")
            continue

        out_images = output_dir / split / "images"
        out_labels = output_dir / split / "labels"

        n_imgs, n_conv, n_skip = convert_split(
            coco_json,
            images_dir=input_dir / split,
            output_images_dir=out_images,
            output_labels_dir=out_labels,
        )

        total_images += n_imgs
        total_converted += n_conv
        total_skipped += n_skip

        warn = ""
        if n_imgs < 10:
            warn = f"  ⚠️  WARN: only {n_imgs} images in {split} split — mAP metrics will be noisy"

        print(
            f"  [{split:5s}] {n_imgs:4d} images | {n_conv:5d} annotations converted | {n_skip:3d} skipped{warn}"
        )

    print()
    print(f"Done! YOLO-seg dataset written to {output_dir}")
    print(f"  Total: {total_images} images, {total_converted} annotations")
    print()
    print("⚠️  VALIDATION SET WARNING:")
    print("  The valid/ split has only 2 images (both from the same video clip).")
    print("  mAP curves during training will be extremely noisy and unreliable.")
    print("  Recommended: re-split the dataset before training.")
    print("  Suggested command to pull ~55 training images into valid:")
    print()
    print("    python -c \"")
    print("    import os, shutil, random")
    print("    from pathlib import Path")
    print("    src = Path('data/processed/court_segmentation_yolo/train')")
    print("    dst = Path('data/processed/court_segmentation_yolo/valid')")
    print("    imgs = list((src / 'images').glob('*'))")
    print("    random.seed(42)")
    print("    val_imgs = random.sample(imgs, 55)")
    print("    for p in val_imgs:")
    print("        shutil.move(str(p), dst / 'images' / p.name)")
    print("        lbl = src / 'labels' / (p.stem + '.txt')")
    print("        if lbl.exists(): shutil.move(str(lbl), dst / 'labels' / lbl.name)")
    print("    print(f'Moved {len(val_imgs)} images to valid split')\"")
    print()
    print("Use configs/court_segmentation.yaml for training with Ultralytics.")


if __name__ == "__main__":
    main()
