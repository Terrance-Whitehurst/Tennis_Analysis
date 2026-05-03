"""
Test trained player detection, court segmentation, and ball detection models
on test video(s).

Runs RF-DETR (player + ball) and RF-DETR-Seg (court instance segmentation)
on each frame, writing an annotated MP4 with semi-transparent court masks,
player ellipses + triangles with ByteTrack IDs, and a tennis-ball trail.

Usage:
    # Default — process every video in test_vids/
    python scripts/test_models_on_video.py

    # Single clip
    python scripts/test_models_on_video.py --video test_vids/Clip1.mp4

    # Skip a model
    python scripts/test_models_on_video.py --no-court
    python scripts/test_models_on_video.py --no-player
    python scripts/test_models_on_video.py --no-ball
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Default paths ────────────────────────────────────────────────────────────
DEFAULT_VIDEO_DIR = "test_vids"
DEFAULT_PLAYER_MODEL = "models/player_detection/checkpoint_best_total.pth"
DEFAULT_COURT_MODEL = "models/court_segmentation/rfdetr_best.pth"
DEFAULT_BALL_MODEL = "models/ball_detection/checkpoint_best_total.pth"
DEFAULT_OUTPUT_DIR = "reports/figures"

# ── Class labels ─────────────────────────────────────────────────────────────
CLASS_NAMES = ["players-balls-court", "player-back", "player-front"]
BALL_CLASS_NAMES = ["tennis-ball", "tennis_ball"]
# Indices match the Roboflow COCO export's category ids (id=0 is the unused
# 'courts' supercategory; the model never predicts it).
COURT_CLASS_NAMES = ["courts", "doubles_alley", "no_mans_land", "service_box"]

# ── Colors ───────────────────────────────────────────────────────────────────
# Court regions — distinct hues, semi-transparent overlays. Index 0 is the
# unused 'courts' supercategory; included only to keep the palette aligned
# with class_id without manual remapping.
COURT_PALETTE = sv.ColorPalette.from_hex([
    "#000000",  # courts (unused)
    "#FFB347",  # doubles_alley — orange
    "#9C6ADE",  # no_mans_land — purple
    "#5DD39E",  # service_box  — mint green
])
PLAYER_COLORS = sv.ColorPalette.from_hex(["#00FF00", "#4488FF"])  # green, blue


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


def load_court_segmentation_model(checkpoint_path, device):
    """Load fine-tuned RF-DETR-Seg-Medium for court instance segmentation."""
    from rfdetr import RFDETRSegMedium

    # Construct with default config, then load our 3-class fine-tuned weights.
    # The checkpoint was written by rfdetr's own trainer, so pretrain_weights=
    # is the documented loading path.
    model = RFDETRSegMedium(pretrain_weights=checkpoint_path)
    model.model.class_names = COURT_CLASS_NAMES
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


def run_court_segmentation(model, frame):
    """Run RF-DETR-Seg court segmentation on a single frame.

    Returns sv.Detections with .mask populated (boolean H×W per instance).
    """
    detections = model.predict(frame, threshold=0.3)
    if detections is None or len(detections) == 0:
        return sv.Detections.empty()
    detections.data.pop("source_image", None)
    detections.data.pop("source_shape", None)
    return detections


def process_video(
    video_path: str,
    output_dir: str,
    player_model,
    court_model,
    ball_model,
    save_frames: bool = False,
):
    """Run all enabled models on a single video and write an annotated MP4."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Could not open video: {video_path}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {video_path}")
    print(f"  Resolution: {w}x{h}, FPS: {fps:.1f}, Frames: {total_frames}")

    stem = Path(video_path).stem
    output_path = os.path.join(output_dir, f"{stem}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # ── Supervision annotators ──────────────────────────────────────────────

    # Court segmentation — polygon outline only (no fill), so player pixels
    # are never tinted by the court overlay colour.
    polygon_annotator = sv.PolygonAnnotator(
        color=COURT_PALETTE,
        thickness=2,
        color_lookup=sv.ColorLookup.CLASS,
    )
    # Court class-name labels — printed at polygon centroid, coloured to match
    # the corresponding outline, white text on coloured background.
    court_label_annotator = sv.LabelAnnotator(
        color=COURT_PALETTE,
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1,
        text_position=sv.Position.CENTER,
        color_lookup=sv.ColorLookup.CLASS,
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

    # Ball detection annotators
    ball_triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#00FFFF"),
        base=20,
        height=16,
        position=sv.Position.TOP_CENTER,
        outline_thickness=2,
        outline_color=sv.Color.BLACK,
    )
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
    total_ball_dets = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # ── Step 1: Player detection + tracking ──
        player_dets_for_frame = None
        if player_model is not None:
            try:
                detections = run_player_detection(player_model, frame)
                if len(detections) > 0:
                    # Track and smooth
                    detections = tracker.update_with_detections(detections)
                    detections = smoother.update_with_detections(detections)
                    player_dets_for_frame = detections
                    total_player_dets += len(detections)

                    if frame_idx % 10 == 0:
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
                        print(
                            f"  Frame {frame_idx:4d} | Players: {len(detections)} [{', '.join(labels)}]"
                        )
                elif frame_idx % 10 == 0:
                    print(f"  Frame {frame_idx:4d} | Players: 0 []")
            except Exception as e:
                print(f"  Player detection error (frame {frame_idx}): {e}")
                raise

        # ── Step 2: Court segmentation — polygon outlines only (no fill) ──
        if court_model is not None:
            try:
                court_dets = run_court_segmentation(court_model, frame)
                if len(court_dets) > 0 and court_dets.mask is not None:
                    annotated = polygon_annotator.annotate(annotated, court_dets)
                    # Add class-name label at each polygon centroid
                    court_labels = [
                        COURT_CLASS_NAMES[int(c)] if int(c) < len(COURT_CLASS_NAMES)
                        else f"class_{int(c)}"
                        for c in court_dets.class_id
                    ]
                    annotated = court_label_annotator.annotate(
                        annotated, court_dets, labels=court_labels
                    )
                    total_court_dets += 1

                    if frame_idx % 10 == 0:
                        names = [
                            COURT_CLASS_NAMES[int(c)] if int(c) < len(COURT_CLASS_NAMES)
                            else f"class_{int(c)}"
                            for c in court_dets.class_id
                        ]
                        print(
                            f"  Frame {frame_idx:4d} | Court regions: "
                            f"{len(court_dets)} [{', '.join(names)}]"
                        )
            except Exception as e:
                if frame_idx == 0:
                    print(f"  Court segmentation error: {e}")

        # ── Step 3: Draw player annotations on top of the court outline ──
        if player_dets_for_frame is not None:
            detections = player_dets_for_frame
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

        # Ball detection with RF-DETR
        if ball_model is not None:
            try:
                ball_detections = run_ball_detection(ball_model, frame)
                if len(ball_detections) > 0:
                    # Take highest confidence detection as "the ball"
                    best_idx = int(np.argmax(ball_detections.confidence))
                    best_det = ball_detections[best_idx : best_idx + 1]
                    # Annotate with supervision (triangle only, no trail)
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

        if save_frames and frame_idx % 30 == 0:
            frame_path = os.path.join(
                output_dir, f"{stem}_frame_{frame_idx:04d}.jpg"
            )
            cv2.imwrite(frame_path, annotated)

        frame_idx += 1

    elapsed = time.time() - start_time
    cap.release()
    out.release()

    print("-" * 60)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Time             : {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps)")
    if player_model is not None:
        print(
            f"  Player dets      : {total_player_dets} "
            f"({total_player_dets / max(frame_idx, 1):.1f}/frame)"
        )
    if court_model is not None:
        print(f"  Court frames     : {total_court_dets}/{frame_idx}")
    if ball_model is not None:
        print(
            f"  Ball frames      : {total_ball_dets}/{frame_idx} "
            f"({total_ball_dets / max(frame_idx, 1) * 100:.1f}%)"
        )
    print(f"  Output           : {output_path}")
    return output_path


def collect_videos(video_arg: str | None) -> list[str]:
    """Resolve --video into a concrete list of files.

    - Explicit file path → that file (must exist).
    - Explicit directory → every .mp4/.mov/.avi inside (sorted).
    - None → DEFAULT_VIDEO_DIR.
    """
    target = Path(video_arg) if video_arg else Path(DEFAULT_VIDEO_DIR)
    if target.is_file():
        return [str(target)]
    if target.is_dir():
        clips = sorted(
            p for p in target.iterdir()
            if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}
        )
        if not clips:
            raise FileNotFoundError(f"No video files found in {target}")
        return [str(p) for p in clips]
    raise FileNotFoundError(f"Video path not found: {target}")


def main():
    parser = argparse.ArgumentParser(description="Run player + ball + court-seg pipeline on video(s)")
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Video file OR directory of clips. "
            f"Default: every video in {DEFAULT_VIDEO_DIR}/"
        ),
    )
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL)
    parser.add_argument("--court-model", default=DEFAULT_COURT_MODEL)
    parser.add_argument("--ball-model", default=DEFAULT_BALL_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-player", action="store_true")
    parser.add_argument("--no-court", action="store_true")
    parser.add_argument("--no-ball", action="store_true")
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also save every 30th annotated frame as JPG",
    )
    args = parser.parse_args()

    videos = collect_videos(args.video)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Videos to process: {len(videos)}")
    for v in videos:
        print(f"  - {v}")

    # ── Load models once, reuse across videos ───────────────────────────────
    player_model = None
    court_model = None
    ball_model = None

    if not args.no_player:
        print(f"\nLoading player detection model: {args.player_model}")
        assert Path(args.player_model).exists(), f"Missing: {args.player_model}"
        player_model = load_player_detection_model(args.player_model, device)
        print("  ✓ Player model loaded.")

    if not args.no_court:
        print(f"Loading court segmentation model: {args.court_model}")
        assert Path(args.court_model).exists(), f"Missing: {args.court_model}"
        court_model = load_court_segmentation_model(args.court_model, device)
        print("  ✓ Court segmentation model loaded.")

    if not args.no_ball:
        print(f"Loading ball detection model: {args.ball_model}")
        assert Path(args.ball_model).exists(), f"Missing: {args.ball_model}"
        ball_model = load_ball_detection_model(args.ball_model, device)
        print("  ✓ Ball model loaded.")

    # ── Process each video ──────────────────────────────────────────────────
    outputs: list[str] = []
    for video in videos:
        print("\n" + "=" * 60)
        out_path = process_video(
            video_path=video,
            output_dir=args.output_dir,
            player_model=player_model,
            court_model=court_model,
            ball_model=ball_model,
            save_frames=args.save_frames,
        )
        outputs.append(out_path)

    print("\n" + "=" * 60)
    print("All videos processed:")
    for p in outputs:
        print(f"  ✓ {p}")


if __name__ == "__main__":
    main()
