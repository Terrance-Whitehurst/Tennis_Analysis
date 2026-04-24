"""
Tennis Ball Tracking using RF-DETR.

Wraps the RF-DETR ball detection model (trained on tennis ball data) to provide
per-frame ball detection and trajectory visualization. No temporal stacking —
each frame is processed independently. The highest-confidence detection per
frame is taken as "the ball"; frames below threshold are marked as no-ball.

Trajectory visualization uses supervision's TraceAnnotator for a fading trail,
matching the annotation style used in test_models_on_video.py.

Usage (CLI):
    python -m src.inference.ball_tracking \
        --video data/raw/test_video/Test_Clip_1.mp4 \
        --rfdetr-model models/ball_detection/checkpoint_best_total.pth \
        --output data/raw/test_video/Video_Output/ball_tracking.mp4
"""

from __future__ import annotations

import argparse
import os
from collections import deque

import cv2
import numpy as np
import numpy.typing as npt
import supervision as sv
import torch
from PIL import Image, ImageDraw

# Default checkpoint path
DEFAULT_BALL_MODEL = "models/ball_detection/checkpoint_best_total.pth"

# Class names for the ball detection model (2 classes + background in head)
BALL_CLASS_NAMES = ["tennis-ball", "tennis_ball"]

# Confidence threshold for ball detection
BALL_THRESHOLD: float = 0.3


def get_device() -> torch.device:
    """Select the best available torch device (MPS > CUDA > CPU).

    Returns:
        The torch device to use for inference.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_ball_detection_model(
    checkpoint_path: str,
    device: torch.device,
):
    """Load RF-DETR ball detection model from a SageMaker training checkpoint.

    Matches the pattern used in scripts/test_models_on_video.py.

    Args:
        checkpoint_path: Path to the checkpoint .pth file.
        device: Torch device to move the model to.

    Returns:
        Loaded RF-DETR model ready for inference.
    """
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


def run_ball_detection(model, frame: npt.NDArray[np.uint8]) -> sv.Detections:
    """Run RF-DETR ball detection on a single frame.

    Args:
        model: Loaded RF-DETR ball detection model.
        frame: BGR frame as a numpy array (H, W, 3).

    Returns:
        sv.Detections with bounding boxes, confidence, and class IDs.
        Returns sv.Detections.empty() if no ball detected above threshold.
    """
    detections = model.predict(frame, threshold=BALL_THRESHOLD)
    if detections is None or len(detections) == 0:
        return sv.Detections.empty()
    detections.data.pop("source_image", None)
    detections.data.pop("source_shape", None)
    return detections


def track_balls(
    video_path: str,
    rfdetr_model_path: str,
) -> dict[str, list]:
    """Run RF-DETR ball detection on every frame of a video.

    Per-frame inference — no temporal stacking. Takes the highest-confidence
    detection per frame as "the ball". Frames with no detection above threshold
    are marked as invisible (Visibility=0, X=0, Y=0).

    Args:
        video_path: Path to the input video file.
        rfdetr_model_path: Path to the RF-DETR ball detection checkpoint.

    Returns:
        Dict with keys 'Frame', 'X', 'Y', 'Visibility', each a list of
        per-frame values in the original video resolution.
    """
    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading ball detection model: {rfdetr_model_path}")
    model = load_ball_detection_model(rfdetr_model_path, device)
    print("Ball detection model loaded.")

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {video_path} — {total_frames} frames, {w}x{h}")

    pred_dict: dict[str, list] = {
        "Frame": [],
        "X": [],
        "Y": [],
        "Visibility": [],
    }

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = run_ball_detection(model, frame)

        if len(detections) > 0:
            # Pick highest-confidence detection as the ball
            best_idx = int(np.argmax(detections.confidence))
            best = detections[best_idx : best_idx + 1]
            cx = int((best.xyxy[0][0] + best.xyxy[0][2]) / 2)
            cy = int((best.xyxy[0][1] + best.xyxy[0][3]) / 2)
            pred_dict["Frame"].append(frame_idx)
            pred_dict["X"].append(cx)
            pred_dict["Y"].append(cy)
            pred_dict["Visibility"].append(1)
        else:
            pred_dict["Frame"].append(frame_idx)
            pred_dict["X"].append(0)
            pred_dict["Y"].append(0)
            pred_dict["Visibility"].append(0)

        if (frame_idx + 1) % 50 == 0:
            detected = sum(pred_dict["Visibility"])
            print(f"  Frame {frame_idx + 1}/{total_frames} | Detections so far: {detected}")

        frame_idx += 1

    cap.release()

    total_detections = sum(pred_dict["Visibility"])
    print(
        f"Done: {frame_idx} frames, {total_detections} ball detections "
        f"({total_detections / max(frame_idx, 1) * 100:.1f}%)"
    )
    return pred_dict


def draw_ball_trajectory(
    frame: npt.NDArray[np.uint8],
    trajectory: deque,
    radius: int = 3,
) -> npt.NDArray[np.uint8]:
    """Draw ball trajectory dots on a frame.

    Renders a fading trail of white circles with yellow outlines for recent
    ball positions.

    Args:
        frame: BGR frame to annotate.
        trajectory: Deque of (x, y) tuples or None for frames with no detection.
        radius: Circle radius in pixels.

    Returns:
        Annotated frame (BGR).
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    for pos in trajectory:
        if pos is not None:
            x, y = pos
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, fill="white", outline="yellow")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def process_video(
    input_path: str,
    output_path: str,
    rfdetr_model_path: str,
    traj_len: int = 8,
) -> None:
    """Run RF-DETR ball tracking and produce an annotated output video.

    Args:
        input_path: Path to the input video file.
        output_path: Path for the annotated output video.
        rfdetr_model_path: Path to the RF-DETR ball detection checkpoint.
        traj_len: Number of past positions to draw as trajectory trail.
    """
    # Run tracking
    pred_dict = track_balls(input_path, rfdetr_model_path)

    # Write output video with trajectory overlay
    print("Writing output video...")
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    trajectory: deque = deque(maxlen=traj_len)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add current position to trajectory deque
        if frame_idx < len(pred_dict["Frame"]) and pred_dict["Visibility"][frame_idx]:
            trajectory.appendleft(
                (pred_dict["X"][frame_idx], pred_dict["Y"][frame_idx])
            )
        else:
            trajectory.appendleft(None)

        annotated = draw_ball_trajectory(frame, trajectory)

        # Overlay info text
        vis_count = sum(pred_dict["Visibility"][: frame_idx + 1])
        ball_status = (
            "DETECTED"
            if frame_idx < len(pred_dict["Visibility"])
            and pred_dict["Visibility"][frame_idx]
            else "MISSING"
        )
        info = (
            f"Frame {frame_idx} | Ball: {ball_status} | "
            f"Detections: {vis_count}/{frame_idx + 1}"
        )
        cv2.putText(
            annotated,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! {frame_idx} frames -> {output_path}")

    # Save CSV
    csv_path = output_path.replace(".mp4", "_ball.csv")
    import pandas as pd

    pd.DataFrame(pred_dict).to_csv(csv_path, index=False)
    print(f"Ball positions saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF-DETR ball tracking inference")
    parser.add_argument(
        "--video",
        default="data/raw/test_video/Test_Clip_1.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--rfdetr-model",
        default=DEFAULT_BALL_MODEL,
        help="Path to RF-DETR ball detection checkpoint",
    )
    parser.add_argument(
        "--output",
        default="data/raw/test_video/Video_Output/ball_tracking.mp4",
        help="Output annotated video path",
    )
    parser.add_argument(
        "--traj-len",
        type=int,
        default=8,
        help="Number of past positions to draw as trajectory trail",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Processing {args.video}...")
    process_video(args.video, args.output, args.rfdetr_model, args.traj_len)
