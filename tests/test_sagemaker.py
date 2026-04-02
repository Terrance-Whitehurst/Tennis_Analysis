"""Tests for SageMaker entry point and launch scripts."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# We test the entry points' pure logic (arg parsing, data conversion)
# without actually launching SageMaker jobs.


class TestPlayerDetectionEntryParsing:
    """Test entry_player_detection.py argument parsing."""

    def test_default_args(self):
        sys.path.insert(0, "scripts/sagemaker")
        from entry_player_detection import parse_args

        with patch("sys.argv", ["entry_player_detection.py"]):
            args = parse_args()
        assert args.model == "base"
        assert args.epochs == 50
        assert args.batch_size == 8
        assert args.lr == 1e-4
        assert args.grad_accum_steps == 4
        assert args.image_size == 560
        assert args.weight_decay == 1e-4
        assert args.warmup_epochs == 0
        assert args.num_workers == 4

    def test_custom_args(self):
        from entry_player_detection import parse_args

        with patch("sys.argv", [
            "entry_player_detection.py",
            "--model", "large",
            "--epochs", "100",
            "--batch_size", "16",
            "--lr", "0.001",
        ]):
            args = parse_args()
        assert args.model == "large"
        assert args.epochs == 100
        assert args.batch_size == 16
        assert args.lr == 0.001

    def test_sagemaker_env_defaults(self):
        from entry_player_detection import parse_args

        env = {
            "SM_MODEL_DIR": "/opt/ml/model",
            "SM_OUTPUT_DATA_DIR": "/opt/ml/output/data",
            "SM_CHANNEL_TRAINING": "/opt/ml/input/data/training",
            "SM_NUM_GPUS": "4",
        }
        with patch.dict(os.environ, env), patch("sys.argv", ["entry_player_detection.py"]):
            args = parse_args()
        assert args.model_dir == "/opt/ml/model"
        assert args.data_dir == "/opt/ml/input/data/training"
        assert args.num_gpus == 4


class TestCourtKeypointEntryParsing:
    """Test entry_court_keypoint.py argument parsing."""

    def test_default_args(self):
        sys.path.insert(0, "scripts/sagemaker")
        from entry_court_keypoint import parse_args

        with patch("sys.argv", ["entry_court_keypoint.py"]):
            args = parse_args()
        assert args.model == "yolo11m-pose.pt"
        assert args.epochs == 100
        assert args.batch == 16
        assert args.imgsz == 640
        assert args.lr0 == 0.01
        assert args.pose == 12.0
        assert args.fliplr == 0.0

    def test_custom_args(self):
        from entry_court_keypoint import parse_args

        with patch("sys.argv", [
            "entry_court_keypoint.py",
            "--model", "yolo11l-pose.pt",
            "--epochs", "200",
            "--batch", "8",
            "--imgsz", "1280",
            "--pose", "15.0",
        ]):
            args = parse_args()
        assert args.model == "yolo11l-pose.pt"
        assert args.epochs == 200
        assert args.batch == 8
        assert args.imgsz == 1280
        assert args.pose == 15.0


class TestCourtKeypointConversion:
    """Test the COCO->YOLO conversion logic in entry_court_keypoint.py."""

    def test_convert_and_prepare_dataset(self):
        from entry_court_keypoint import convert_coco_to_yolo_kpt, prepare_yolo_dataset
        from PIL import Image as PILImage

        with tempfile.TemporaryDirectory() as tmpdir:
            coco_dir = Path(tmpdir) / "coco"
            # Create train split
            train_dir = coco_dir / "train"
            train_dir.mkdir(parents=True)

            img = PILImage.new("RGB", (640, 480))
            img.save(str(train_dir / "img_0000.jpg"))

            # Create minimal COCO JSON
            keypoints = []
            for _ in range(14):
                keypoints.extend([100.0, 100.0, 2])

            coco_data = {
                "images": [{"id": 0, "file_name": "img_0000.jpg", "width": 640, "height": 480}],
                "annotations": [{
                    "id": 0, "image_id": 0, "category_id": 1,
                    "bbox": [10, 10, 200, 150],
                    "keypoints": keypoints,
                }],
                "categories": [{"id": 1, "name": "tenniscourt"}],
            }
            with open(str(train_dir / "_annotations.coco.json"), "w") as f:
                json.dump(coco_data, f)

            # Convert
            yolo_dir = Path(tmpdir) / "yolo"
            yaml_path = prepare_yolo_dataset(str(coco_dir), str(yolo_dir))

            assert os.path.exists(yaml_path)
            assert (yolo_dir / "train" / "labels").exists()
            label_files = list((yolo_dir / "train" / "labels").glob("*.txt"))
            assert len(label_files) == 1


def _mock_sagemaker_imports():
    """Mock sagemaker.pytorch so launch scripts can be imported without full SDK."""
    mock_pytorch = MagicMock()
    sys.modules.setdefault("sagemaker", MagicMock())
    sys.modules.setdefault("sagemaker.pytorch", mock_pytorch)


class TestLaunchPlayerDetectionParsing:
    """Test launch_player_detection.py argument parsing."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _mock_sagemaker_imports()
        # Force reimport with mocked modules
        if "launch_player_detection" in sys.modules:
            del sys.modules["launch_player_detection"]

    def test_requires_role(self):
        sys.path.insert(0, "scripts/sagemaker")
        from launch_player_detection import parse_args

        with patch("sys.argv", ["launch_player_detection.py"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_default_hyperparams(self):
        from launch_player_detection import parse_args

        with patch("sys.argv", [
            "launch_player_detection.py",
            "--role", "arn:aws:iam::123456789012:role/TestRole",
        ]):
            args = parse_args()
        assert args.role == "arn:aws:iam::123456789012:role/TestRole"
        assert args.instance_type == "ml.g4dn.xlarge"
        assert args.model == "base"
        assert args.epochs == 50
        assert args.spot is False
        assert args.wait is False

    def test_spot_instance_flag(self):
        from launch_player_detection import parse_args

        with patch("sys.argv", [
            "launch_player_detection.py",
            "--role", "arn:aws:iam::123456789012:role/TestRole",
            "--spot",
        ]):
            args = parse_args()
        assert args.spot is True


class TestLaunchCourtKeypointParsing:
    """Test launch_court_keypoint.py argument parsing."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        _mock_sagemaker_imports()
        if "launch_court_keypoint" in sys.modules:
            del sys.modules["launch_court_keypoint"]

    def test_default_hyperparams(self):
        sys.path.insert(0, "scripts/sagemaker")
        from launch_court_keypoint import parse_args

        with patch("sys.argv", [
            "launch_court_keypoint.py",
            "--role", "arn:aws:iam::123456789012:role/TestRole",
        ]):
            args = parse_args()
        assert args.model == "yolo11m-pose.pt"
        assert args.epochs == 100
        assert args.batch == 16
        assert args.instance_type == "ml.g4dn.xlarge"

    def test_tags_parsing(self):
        from launch_court_keypoint import parse_args

        with patch("sys.argv", [
            "launch_court_keypoint.py",
            "--role", "arn:aws:iam::123456789012:role/TestRole",
            "--tags", "team=cv", "env=dev",
        ]):
            args = parse_args()
        assert args.tags == ["team=cv", "env=dev"]
