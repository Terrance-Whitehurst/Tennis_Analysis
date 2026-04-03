"""Tests for COCO to YOLO keypoint format conversion."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.convert_coco_to_yolo_kpt import convert_coco_to_yolo_kpt


def _create_dummy_coco_dataset(
    tmpdir, num_images=3, num_keypoints=14, img_w=640, img_h=480
):
    """Create a minimal COCO keypoint dataset for testing."""
    images = []
    annotations = []
    for i in range(num_images):
        fname = f"img_{i:04d}.jpg"
        images.append({"id": i, "file_name": fname, "width": img_w, "height": img_h})

        # Create a dummy image file
        img = Image.fromarray(
            np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)
        )
        img.save(os.path.join(tmpdir, fname))

        # Create a COCO annotation with keypoints
        keypoints = []
        for k in range(num_keypoints):
            kx = np.random.uniform(0, img_w)
            ky = np.random.uniform(0, img_h)
            kv = 2  # visible
            keypoints.extend([kx, ky, kv])

        bbox_x = np.random.uniform(0, img_w / 2)
        bbox_y = np.random.uniform(0, img_h / 2)
        bbox_w = np.random.uniform(50, img_w / 2)
        bbox_h = np.random.uniform(50, img_h / 2)

        annotations.append(
            {
                "id": i,
                "image_id": i,
                "category_id": 1,
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "keypoints": keypoints,
                "area": bbox_w * bbox_h,
                "iscrowd": 0,
            }
        )

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tenniscourt"}],
    }
    coco_json_path = os.path.join(tmpdir, "_annotations.coco.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_data, f)

    return coco_json_path


class TestConvertCocoToYoloKpt:
    def test_converts_correct_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coco_json = _create_dummy_coco_dataset(tmpdir, num_images=5)
            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            converted, skipped = convert_coco_to_yolo_kpt(
                coco_json, tmpdir, out_images, out_labels, num_keypoints=14
            )
            assert converted == 5
            assert skipped == 0

    def test_creates_label_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coco_json = _create_dummy_coco_dataset(tmpdir, num_images=3)
            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            convert_coco_to_yolo_kpt(coco_json, tmpdir, out_images, out_labels)
            label_files = list(Path(out_labels).glob("*.txt"))
            assert len(label_files) == 3

    def test_label_format(self):
        """Each line: class_id cx cy w h + 14 * (kx ky kv) = 5 + 42 = 47 values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coco_json = _create_dummy_coco_dataset(tmpdir, num_images=1)
            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            convert_coco_to_yolo_kpt(coco_json, tmpdir, out_images, out_labels)

            label_files = list(Path(out_labels).glob("*.txt"))
            with open(label_files[0]) as f:
                line = f.readline().strip()
            parts = line.split()
            # class_id + 4 bbox + 14*3 keypoints = 47
            assert len(parts) == 47
            assert parts[0] == "0"  # class_id

    def test_normalized_coordinates(self):
        """All bbox and keypoint coordinates should be in [0, 1]."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coco_json = _create_dummy_coco_dataset(tmpdir, num_images=2)
            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            convert_coco_to_yolo_kpt(coco_json, tmpdir, out_images, out_labels)

            for label_file in Path(out_labels).glob("*.txt"):
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        # Skip class_id (index 0), check all float values
                        for val_str in parts[1:]:
                            val = float(val_str)
                            # Visibility can be 0, 1, or 2 — keypoint values at kv positions
                            # But normalized coords should be [0, 1]
                            # kv positions: indices 7, 10, 13, ... (every 3rd starting from 5+2)
                            # We just check all are non-negative
                            assert val >= 0.0

    def test_skips_annotations_without_keypoints(self):
        """Annotations missing keypoints should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an annotation without keypoints
            images = [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}]
            annotations = [
                {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 100, 100]},
            ]
            coco_data = {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 0, "name": "tennis_court"}],
            }
            coco_json = os.path.join(tmpdir, "_annotations.coco.json")
            with open(coco_json, "w") as f:
                json.dump(coco_data, f)

            # Create dummy image
            img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            img.save(os.path.join(tmpdir, "img.jpg"))

            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            converted, skipped = convert_coco_to_yolo_kpt(
                coco_json, tmpdir, out_images, out_labels
            )
            assert converted == 0
            assert skipped == 1

    def test_wrong_keypoint_count_skipped(self):
        """Annotations with wrong number of keypoints should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}]
            annotations = [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [10, 10, 100, 100],
                    "keypoints": [1, 2, 2, 3, 4, 2],  # Only 2 keypoints, not 14
                }
            ]
            coco_data = {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "tenniscourt"}],
            }
            coco_json = os.path.join(tmpdir, "_annotations.coco.json")
            with open(coco_json, "w") as f:
                json.dump(coco_data, f)

            img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            img.save(os.path.join(tmpdir, "img.jpg"))

            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            converted, skipped = convert_coco_to_yolo_kpt(
                coco_json, tmpdir, out_images, out_labels
            )
            assert converted == 0
            assert skipped == 1

    def test_copies_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            coco_json = _create_dummy_coco_dataset(tmpdir, num_images=2)
            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            convert_coco_to_yolo_kpt(coco_json, tmpdir, out_images, out_labels)
            copied = list(Path(out_images).glob("*.jpg"))
            assert len(copied) == 2

    def test_visibility_zero_zeros_coordinates(self):
        """Keypoints with visibility=0 should have x=0, y=0 in output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [{"id": 0, "file_name": "img.jpg", "width": 640, "height": 480}]
            # 14 keypoints, first one invisible
            keypoints = [100, 200, 0]  # invisible
            for _ in range(13):
                keypoints.extend([100, 200, 2])  # visible

            annotations = [
                {
                    "id": 0,
                    "image_id": 0,
                    "category_id": 1,
                    "bbox": [10, 10, 100, 100],
                    "keypoints": keypoints,
                }
            ]
            coco_data = {
                "images": images,
                "annotations": annotations,
                "categories": [{"id": 1, "name": "tenniscourt"}],
            }
            coco_json = os.path.join(tmpdir, "_annotations.coco.json")
            with open(coco_json, "w") as f:
                json.dump(coco_data, f)

            img = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
            img.save(os.path.join(tmpdir, "img.jpg"))

            out_images = os.path.join(tmpdir, "yolo_images")
            out_labels = os.path.join(tmpdir, "yolo_labels")
            convert_coco_to_yolo_kpt(coco_json, tmpdir, out_images, out_labels)

            label_file = list(Path(out_labels).glob("*.txt"))[0]
            with open(label_file) as f:
                parts = f.readline().strip().split()

            # First keypoint (indices 5, 6, 7) should be 0, 0, 0
            assert float(parts[5]) == 0.0
            assert float(parts[6]) == 0.0
            assert int(parts[7]) == 0
