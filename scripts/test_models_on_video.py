"""
Test trained player detection and court keypoint models on test video.

Runs RF-DETR (player detection) and YOLO-Pose (court keypoint detection)
on each frame of the test video, saves an annotated output video, and
prints per-frame detection summaries.

Usage:
    python scripts/test_models_on_video.py
    python scripts/test_models_on_video.py --video data/raw/test_video/Test_Clip_1.mp4
    python scripts/test_models_on_video.py --no-court  # skip court keypoint model
    python scripts/test_models_on_video.py --no-player  # skip player detection model
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Default paths
DEFAULT_VIDEO = "data/raw/test_video/Test_Clip_1.mp4"
DEFAULT_PLAYER_MODEL = "models/player_detection/checkpoint_best_total.pth"
DEFAULT_COURT_MODEL = "models/court_keypoint/best.pt"
DEFAULT_OUTPUT_DIR = "reports/figures"

# Court keypoint skeleton connections — must match configs/court_keypoint.yaml exactly
# These trace the actual court lines (baselines, sidelines, service lines, center line)
SKELETON = [
    (0, 4),  # corner1 -> service_line1
    (4, 6),  # service_line1 -> service_line3
    (6, 1),  # service_line3 -> corner2
    (1, 3),  # corner2 -> corner4
    (3, 7),  # corner4 -> service_line4
    (7, 5),  # service_line4 -> service_line2
    (5, 2),  # service_line2 -> corner3
    (2, 0),  # corner3 -> corner1
    (4, 8),  # service_line1 -> inner1
    (8, 10),  # inner1 -> inner3
    (10, 5),  # inner3 -> service_line2
    (6, 9),  # service_line3 -> inner2
    (9, 11),  # inner2 -> inner4
    (11, 7),  # inner4 -> service_line4
    (8, 12),  # inner1 -> center1
    (12, 9),  # center1 -> inner2
    (10, 13),  # inner3 -> center2
    (13, 11),  # center2 -> inner4
    (12, 13),  # center1 -> center2 (center service line)
]

# Colors
PLAYER_COLORS = {
    "player-back": (0, 255, 0),  # green
    "player-front": (0, 0, 255),  # red
}
COURT_COLOR = (0, 0, 255)  # red (BGR) — matches reference style
KEYPOINT_COLOR = (0, 0, 255)  # red


CLASS_NAMES = ["players-balls-court", "player-back", "player-front"]


def load_player_detection_model(checkpoint_path, device):
    """Load RF-DETR model from SageMaker training checkpoint."""
    from rfdetr import RFDETRBase

    model = RFDETRBase()
    # Reinitialize detection head for 4 outputs (3 classes + background index)
    model.model.reinitialize_detection_head(num_classes=4)
    model.model.class_names = CLASS_NAMES

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    # Strip the 'model.' prefix added by PyTorch Lightning
    cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.model.model.load_state_dict(cleaned)
    model.model.model.to(device)
    model.model.device = device
    model.model.model.eval()
    return model


def load_court_keypoint_model(checkpoint_path):
    """Load YOLO-Pose model for court keypoint detection."""
    from ultralytics import YOLO

    model = YOLO(checkpoint_path)
    return model


def run_player_detection(model, frame):
    """Run RF-DETR player detection on a single frame.

    Returns list of dicts with keys: bbox, confidence, class_name.
    Uses supervision (sv.Detections) returned by rfdetr.predict().
    """
    detections = model.predict(frame, threshold=0.5)
    results = []
    if detections is None or len(detections) == 0:
        return results

    for i in range(len(detections)):
        bbox = detections.xyxy[i].tolist()
        conf = detections.confidence[i].item()
        cls_id = int(detections.class_id[i])
        class_name = (
            CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        )
        results.append(
            {
                "bbox": bbox,
                "confidence": conf,
                "class_name": class_name,
            }
        )
    return results


def run_court_keypoint(model, frame):
    """Run YOLO-Pose court keypoint detection on a single frame.

    Returns list of results from ultralytics.
    """
    results = model.predict(frame, verbose=False)
    return results


def draw_player_detections(frame, detections):
    """Draw bounding boxes and labels for player detections."""
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        class_name = det["class_name"]
        conf = det["confidence"]
        color = PLAYER_COLORS.get(class_name, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 6),
            (x1 + label_size[0], y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    return frame


def draw_court_keypoints(frame, results):
    """Draw court lines by connecting detected keypoints with the skeleton.

    Draws clean red lines tracing the actual court geometry (baselines,
    sidelines, service lines, center service line) — no bounding boxes,
    no keypoint labels.
    """
    if not results or len(results) == 0:
        return frame

    result = results[0]
    if result.keypoints is None or len(result.keypoints) == 0:
        return frame

    keypoints = result.keypoints.xy.cpu().numpy()
    if len(keypoints) == 0:
        return frame

    for kpts in keypoints:
        # Draw skeleton lines (court wireframe)
        for i, j in SKELETON:
            if i < len(kpts) and j < len(kpts):
                x1, y1 = kpts[i]
                x2, y2 = kpts[j]
                if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                    cv2.line(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        COURT_COLOR,
                        2,
                    )

    return frame


def main():
    parser = argparse.ArgumentParser(description="Test models on video")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path")
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL)
    parser.add_argument("--court-model", default=DEFAULT_COURT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-player", action="store_true", help="Skip player detection"
    )
    parser.add_argument(
        "--no-court", action="store_true", help="Skip court keypoint detection"
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Save individual annotated frames"
    )
    args = parser.parse_args()

    # Validate inputs
    assert Path(args.video).exists(), f"Video not found: {args.video}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load models
    player_model = None
    court_model = None

    if not args.no_player:
        print(f"Loading player detection model: {args.player_model}")
        assert Path(args.player_model).exists(), (
            f"Player model not found: {args.player_model}"
        )
        player_model = load_player_detection_model(args.player_model, device)
        print("  Player detection model loaded.")

    if not args.no_court:
        print(f"Loading court keypoint model: {args.court_model}")
        assert Path(args.court_model).exists(), (
            f"Court model not found: {args.court_model}"
        )
        court_model = load_court_keypoint_model(args.court_model)
        print("  Court keypoint model loaded.")

    # Open video
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Could not open video: {args.video}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {args.video}")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.1f}, Frames: {total_frames}")

    # Output video writer
    output_path = os.path.join(args.output_dir, "test_inference_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"\nRunning inference on {total_frames} frames...")
    print("-" * 60)

    frame_idx = 0
    total_player_dets = 0
    total_court_dets = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # Player detection
        if player_model is not None:
            try:
                player_dets = run_player_detection(player_model, frame)
                annotated = draw_player_detections(annotated, player_dets)
                total_player_dets += len(player_dets)
                if frame_idx % 10 == 0:
                    det_summary = ", ".join(
                        f"{d['class_name']}({d['confidence']:.2f})" for d in player_dets
                    )
                    print(
                        f"  Frame {frame_idx:4d} | Players: {len(player_dets)} [{det_summary}]"
                    )
            except Exception as e:
                if frame_idx == 0:
                    print(f"  Player detection error: {e}")
                    print("  Trying alternative loading approach...")
                    # RF-DETR from roboflow may use a different predict API
                    player_model = None

        # Court keypoint detection
        if court_model is not None:
            try:
                court_results = run_court_keypoint(court_model, frame)
                annotated = draw_court_keypoints(annotated, court_results)
                if (
                    court_results
                    and len(court_results) > 0
                    and court_results[0].boxes is not None
                ):
                    total_court_dets += len(court_results[0].boxes)
                if frame_idx % 10 == 0 and court_results:
                    r = court_results[0]
                    n_kpts = 0
                    if r.keypoints is not None and len(r.keypoints) > 0:
                        kpts = r.keypoints.xy.cpu().numpy()
                        if len(kpts) > 0:
                            n_kpts = np.sum((kpts[0][:, 0] > 0) & (kpts[0][:, 1] > 0))
                    print(
                        f"  Frame {frame_idx:4d} | Court keypoints detected: {n_kpts}/14"
                    )
            except Exception as e:
                if frame_idx == 0:
                    print(f"  Court keypoint error: {e}")

        # Write frame
        out.write(annotated)

        # Save individual frames if requested
        if args.save_frames and frame_idx % 30 == 0:
            frame_path = os.path.join(args.output_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, annotated)

        frame_idx += 1

    elapsed = time.time() - start_time
    cap.release()
    out.release()

    print("-" * 60)
    print("\nResults:")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Time: {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")
    if player_model is not None:
        print(
            f"  Total player detections: {total_player_dets} ({total_player_dets / max(frame_idx, 1):.1f} avg/frame)"
        )
    if court_model is not None:
        print(f"  Total court detections: {total_court_dets}")
    print(f"\n  Output video: {output_path}")


if __name__ == "__main__":
    main()
