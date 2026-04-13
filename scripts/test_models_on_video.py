"""
Test trained player detection, court keypoint, scoreboard detection, and
ball detection models on test video.

Runs RF-DETR (player detection + scoreboard detection + ball detection),
YOLO-Pose (court keypoint detection) on each frame of the test video,
saves an annotated output video with supervision-powered visualizations:
smooth player trails, clean court wireframe, scoreboard bounding boxes,
ball detection with trajectory trail, and polished annotations.

Usage:
    python scripts/test_models_on_video.py
    python scripts/test_models_on_video.py --video data/raw/test_video/Test_Clip_1.mp4
    python scripts/test_models_on_video.py --no-court       # skip court keypoint model
    python scripts/test_models_on_video.py --no-player      # skip player detection model
    python scripts/test_models_on_video.py --no-scoreboard  # skip scoreboard detection model
    python scripts/test_models_on_video.py --no-ball        # skip ball detection model
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Default paths
DEFAULT_VIDEO = "data/raw/test_video/Test_Clip_1.mp4"
DEFAULT_PLAYER_MODEL = "models/player_detection/checkpoint_best_total.pth"
DEFAULT_COURT_MODEL = "models/court_keypoint/best.pt"
DEFAULT_SCOREBOARD_MODEL = "models/scoreboard_detection/checkpoint_best_total.pth"
DEFAULT_BALL_MODEL = "models/ball_detection/checkpoint_best_total.pth"
DEFAULT_OUTPUT_DIR = "reports/figures"

# Court keypoint skeleton connections — 1-indexed to match supervision's EdgeAnnotator
# (sv.EdgeAnnotator internally does xy[index - 1])
# These trace the actual court lines (baselines, sidelines, service lines, center line)
SKELETON = [
    (1, 5),  # corner1 -> service_line1
    (5, 7),  # service_line1 -> service_line3
    (7, 2),  # service_line3 -> corner2
    (2, 4),  # corner2 -> corner4
    (4, 8),  # corner4 -> service_line4
    (8, 6),  # service_line4 -> service_line2
    (6, 3),  # service_line2 -> corner3
    (3, 1),  # corner3 -> corner1
    (5, 9),  # service_line1 -> inner1
    (9, 11),  # inner1 -> inner3
    (11, 6),  # inner3 -> service_line2
    (7, 10),  # service_line3 -> inner2
    (10, 12),  # inner2 -> inner4
    (12, 8),  # inner4 -> service_line4
    (9, 13),  # inner1 -> center1
    (13, 10),  # center1 -> inner2
    (11, 14),  # inner3 -> center2
    (14, 12),  # center2 -> inner4
    (13, 14),  # center1 -> center2 (center service line)
]

CLASS_NAMES = ["players-balls-court", "player-back", "player-front"]
SCOREBOARD_CLASS_NAMES = ["scoreboards"]
BALL_CLASS_NAMES = ["tennis-ball", "tennis_ball"]

# Colors
COURT_LINE_COLOR = sv.Color.from_hex("#FF4444")  # red
COURT_VERTEX_COLOR = sv.Color.from_hex("#FF6666")
PLAYER_COLORS = sv.ColorPalette.from_hex(["#00FF00", "#4488FF"])  # green, blue
SCOREBOARD_COLOR = sv.Color.from_hex("#FFD700")  # gold


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


def load_scoreboard_detection_model(checkpoint_path, device):
    """Load RF-DETR model from SageMaker training checkpoint for scoreboard detection."""
    from rfdetr import RFDETRBase

    model = RFDETRBase()
    # Reinitialize detection head for 3 outputs (2 classes + background index)
    model.model.reinitialize_detection_head(num_classes=3)
    model.model.class_names = SCOREBOARD_CLASS_NAMES

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    # Strip the 'model.' prefix added by PyTorch Lightning
    cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.model.model.load_state_dict(cleaned)
    model.model.model.to(device)
    model.model.device = device
    model.model.model.eval()
    return model


def run_player_detection(model, frame):
    """Run RF-DETR player detection on a single frame.

    Returns sv.Detections with bounding boxes, confidence, and class IDs.
    Strips source_image/source_shape from data to avoid indexing errors
    with ByteTrack.
    """
    detections = model.predict(frame, threshold=0.5)
    if detections is None or len(detections) == 0:
        return sv.Detections.empty()
    # Remove non-indexable arrays that break ByteTrack filtering
    detections.data.pop("source_image", None)
    detections.data.pop("source_shape", None)
    return detections


def run_scoreboard_detection(model, frame):
    """Run RF-DETR scoreboard detection on a single frame.

    Returns sv.Detections with bounding boxes, confidence, and class IDs.
    """
    detections = model.predict(frame, threshold=0.5)
    if detections is None or len(detections) == 0:
        return sv.Detections.empty()
    detections.data.pop("source_image", None)
    detections.data.pop("source_shape", None)
    return detections


def load_ball_detection_model(checkpoint_path, device):
    """Load RF-DETR model from SageMaker training checkpoint for ball detection."""
    from rfdetr import RFDETRBase

    model = RFDETRBase()
    # Reinitialize detection head for 3 outputs (2 classes + background index)
    model.model.reinitialize_detection_head(num_classes=3)
    model.model.class_names = BALL_CLASS_NAMES

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    # Strip the 'model.' prefix added by PyTorch Lightning
    cleaned = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    model.model.model.load_state_dict(cleaned)
    model.model.model.to(device)
    model.model.device = device
    model.model.model.eval()
    return model


def run_ball_detection(model, frame):
    """Run RF-DETR ball detection on a single frame.

    Returns sv.Detections with bounding boxes, confidence, and class IDs.
    """
    detections = model.predict(frame, threshold=0.3)
    if detections is None or len(detections) == 0:
        return sv.Detections.empty()
    detections.data.pop("source_image", None)
    detections.data.pop("source_shape", None)
    return detections


def run_court_keypoint(model, frame):
    """Run YOLO-Pose court keypoint detection on a single frame.

    Returns sv.KeyPoints for use with supervision annotators.
    """
    results = model.predict(frame, verbose=False)
    if not results or len(results) == 0:
        return None, None
    r = results[0]
    keypoints = sv.KeyPoints.from_ultralytics(r)
    return keypoints, r


def main():
    parser = argparse.ArgumentParser(description="Test models on video")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path")
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL)
    parser.add_argument("--court-model", default=DEFAULT_COURT_MODEL)
    parser.add_argument("--scoreboard-model", default=DEFAULT_SCOREBOARD_MODEL)
    parser.add_argument("--ball-model", default=DEFAULT_BALL_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-player", action="store_true", help="Skip player detection"
    )
    parser.add_argument(
        "--no-court", action="store_true", help="Skip court keypoint detection"
    )
    parser.add_argument(
        "--no-scoreboard", action="store_true", help="Skip scoreboard detection"
    )
    parser.add_argument("--no-ball", action="store_true", help="Skip ball detection")
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
    scoreboard_model = None

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

    if not args.no_scoreboard:
        print(f"Loading scoreboard detection model: {args.scoreboard_model}")
        assert Path(args.scoreboard_model).exists(), (
            f"Scoreboard model not found: {args.scoreboard_model}"
        )
        scoreboard_model = load_scoreboard_detection_model(
            args.scoreboard_model, device
        )
        print("  Scoreboard detection model loaded.")

    # Load ball detection model (RF-DETR, runs per-frame like other detectors)
    ball_model = None
    if not args.no_ball:
        print(f"Loading ball detection model: {args.ball_model}")
        assert Path(args.ball_model).exists(), (
            f"Ball model not found: {args.ball_model}"
        )
        ball_model = load_ball_detection_model(args.ball_model, device)
        print("  Ball detection model loaded.")

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

    # --- Supervision annotators ---

    # Court keypoint annotators
    edge_annotator = sv.EdgeAnnotator(
        color=COURT_LINE_COLOR,
        thickness=2,
        edges=SKELETON,
    )
    vertex_annotator = sv.VertexAnnotator(
        color=COURT_VERTEX_COLOR,
        radius=4,
    )

    # Player detection annotators
    triangle_annotator = sv.TriangleAnnotator(
        color=PLAYER_COLORS,
        base=30,
        height=25,
        position=sv.Position.TOP_CENTER,
        color_lookup=sv.ColorLookup.TRACK,
        outline_thickness=2,
        outline_color=sv.Color.BLACK,
    )
    ellipse_annotator = sv.EllipseAnnotator(
        color=PLAYER_COLORS,
        thickness=2,
        start_angle=-45,
        end_angle=235,
        color_lookup=sv.ColorLookup.TRACK,
    )
    # Scoreboard detection annotators
    scoreboard_box_annotator = sv.BoxCornerAnnotator(
        color=sv.ColorPalette.from_hex(["#FFD700"]),
        thickness=2,
        corner_length=15,
    )
    scoreboard_label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#FFD700"]),
        text_color=sv.Color.BLACK,
        text_scale=0.5,
        text_padding=4,
    )

    # Ball detection annotators
    ball_triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#00FFFF"),
        base=20,
        height=16,
        position=sv.Position.TOP_CENTER,
        outline_thickness=2,
        outline_color=sv.Color.BLACK,
    )
    ball_trace_annotator = sv.TraceAnnotator(
        color=sv.ColorPalette.from_hex(["#00FFFF"]),
        position=sv.Position.CENTER,
        trace_length=int(fps * 1.5),
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,
    )
    ball_smoother = sv.DetectionsSmoother(length=5)

    # Tracker and smoother for temporal consistency
    tracker = sv.ByteTrack(
        track_activation_threshold=0.4,
        lost_track_buffer=int(fps),  # keep track alive for 1 second
        minimum_matching_threshold=0.8,
        frame_rate=int(fps),
    )
    smoother = sv.DetectionsSmoother(length=5)

    print(f"\nRunning inference on {total_frames} frames...")
    print("-" * 60)

    frame_idx = 0
    total_player_dets = 0
    total_court_dets = 0
    total_scoreboard_dets = 0
    total_ball_dets = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # Court keypoint detection (draw first, underneath player annotations)
        if court_model is not None:
            try:
                keypoints, raw_result = run_court_keypoint(court_model, frame)
                if keypoints is not None and len(keypoints) > 0:
                    annotated = edge_annotator.annotate(annotated, keypoints)
                    annotated = vertex_annotator.annotate(annotated, keypoints)
                    total_court_dets += 1

                    if frame_idx % 10 == 0:
                        n_kpts = 0
                        if keypoints.xy is not None and len(keypoints.xy) > 0:
                            kpts = keypoints.xy[0]
                            n_kpts = np.sum((kpts[:, 0] > 0) & (kpts[:, 1] > 0))
                        print(f"  Frame {frame_idx:4d} | Court keypoints: {n_kpts}/14")
            except Exception as e:
                if frame_idx == 0:
                    print(f"  Court keypoint error: {e}")

        # Scoreboard detection
        if scoreboard_model is not None:
            try:
                sb_detections = run_scoreboard_detection(scoreboard_model, frame)
                if len(sb_detections) > 0:
                    sb_labels = []
                    for i in range(len(sb_detections)):
                        conf = sb_detections.confidence[i]
                        sb_labels.append(f"scoreboard {conf:.2f}")

                    annotated = scoreboard_box_annotator.annotate(
                        annotated, sb_detections
                    )
                    annotated = scoreboard_label_annotator.annotate(
                        annotated, sb_detections, labels=sb_labels
                    )
                    total_scoreboard_dets += len(sb_detections)

                    if frame_idx % 10 == 0:
                        print(
                            f"  Frame {frame_idx:4d} | Scoreboards: {len(sb_detections)}"
                        )
            except Exception as e:
                if frame_idx == 0:
                    print(f"  Scoreboard detection error: {e}")

        # Player detection with tracking
        if player_model is not None:
            try:
                detections = run_player_detection(player_model, frame)
                if len(detections) > 0:
                    # Track and smooth
                    detections = tracker.update_with_detections(detections)
                    detections = smoother.update_with_detections(detections)

                    # Build labels
                    labels = []
                    for i in range(len(detections)):
                        cls_id = int(detections.class_id[i])
                        conf = detections.confidence[i]
                        name = (
                            CLASS_NAMES[cls_id]
                            if cls_id < len(CLASS_NAMES)
                            else f"class_{cls_id}"
                        )
                        labels.append(f"{name} {conf:.2f}")

                    # Annotate: trace + ellipse (narrowed) + semi-transparent triangle
                    # Shrink boxes horizontally for a tighter ellipse
                    narrow_xyxy = detections.xyxy.copy()
                    cx = (narrow_xyxy[:, 0] + narrow_xyxy[:, 2]) / 2
                    half_w = (narrow_xyxy[:, 2] - narrow_xyxy[:, 0]) * 0.35
                    narrow_xyxy[:, 0] = cx - half_w
                    narrow_xyxy[:, 2] = cx + half_w
                    narrow = sv.Detections(
                        xyxy=narrow_xyxy,
                        confidence=detections.confidence,
                        class_id=detections.class_id,
                        tracker_id=detections.tracker_id,
                    )
                    annotated = ellipse_annotator.annotate(annotated, narrow)
                    # Semi-transparent triangle
                    overlay = annotated.copy()
                    overlay = triangle_annotator.annotate(overlay, detections)
                    cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

                    total_player_dets += len(detections)

                    if frame_idx % 10 == 0:
                        det_summary = ", ".join(labels)
                        print(
                            f"  Frame {frame_idx:4d} | Players: {len(detections)} [{det_summary}]"
                        )
                elif frame_idx % 10 == 0:
                    print(f"  Frame {frame_idx:4d} | Players: 0 []")
            except Exception as e:
                print(f"  Player detection error (frame {frame_idx}): {e}")
                raise

        # Ball detection with RF-DETR
        if ball_model is not None:
            try:
                ball_detections = run_ball_detection(ball_model, frame)
                if len(ball_detections) > 0:
                    # Take highest confidence detection as "the ball"
                    best_idx = int(np.argmax(ball_detections.confidence))
                    best_det = ball_detections[best_idx : best_idx + 1]
                    # Assign a stable tracker_id so TraceAnnotator can draw a trail
                    best_det.tracker_id = np.array([0])

                    # Smooth and annotate with supervision
                    best_det = ball_smoother.update_with_detections(best_det)
                    annotated = ball_trace_annotator.annotate(annotated, best_det)
                    annotated = ball_triangle_annotator.annotate(annotated, best_det)
                    total_ball_dets += 1

                    if frame_idx % 10 == 0:
                        cx = int((best_det.xyxy[0][0] + best_det.xyxy[0][2]) / 2)
                        cy = int((best_det.xyxy[0][1] + best_det.xyxy[0][3]) / 2)
                        print(
                            f"  Frame {frame_idx:4d} | Ball: ({cx}, {cy}) conf={best_det.confidence[0]:.2f}"
                        )
            except Exception as e:
                if frame_idx == 0:
                    print(f"  Ball detection error: {e}")

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
            f"  Total player detections: {total_player_dets}"
            f" ({total_player_dets / max(frame_idx, 1):.1f} avg/frame)"
        )
    if court_model is not None:
        print(f"  Court detections: {total_court_dets}/{frame_idx} frames")
    if scoreboard_model is not None:
        print(
            f"  Total scoreboard detections: {total_scoreboard_dets}"
            f" ({total_scoreboard_dets / max(frame_idx, 1):.1f} avg/frame)"
        )
    if ball_model is not None:
        print(
            f"  Ball detections: {total_ball_dets}/{frame_idx} frames"
            f" ({total_ball_dets / max(frame_idx, 1) * 100:.1f}%)"
        )
    print(f"\n  Output video: {output_path}")


if __name__ == "__main__":
    main()
