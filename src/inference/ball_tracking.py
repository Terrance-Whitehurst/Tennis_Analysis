"""
Tennis Ball Tracking using TrackNetV3.

Uses the pretrained TrackNetV3 model (U-Net encoder-decoder) to detect and
track the tennis ball across video frames. The model takes sequences of 8
consecutive frames (+ a median background image), produces per-frame heatmaps,
and extracts ball coordinates from the largest contour in each heatmap.

An optional InpaintNet module fills in gaps where the ball was occluded or
missed by TrackNet, using surrounding trajectory context.

Architecture:
    TrackNet: 2D U-Net (Double/Triple Conv blocks, 3 pool/upsample stages)
        Input: (N, 27, 288, 512) — 8 RGB frames + 1 background, channel-concat
        Output: (N, 8, 288, 512) — per-frame heatmaps via sigmoid

    InpaintNet: 1D U-Net operating on coordinate sequences
        Input: (N, L, 3) — normalized (x, y) + inpaint mask
        Output: (N, L, 2) — refined (x, y) coordinates

Reference:
    qaz812345/TrackNetV3 — "Enhancing ShuttleCock Tracking with Augmentations
    and Trajectory Rectification" (ACM 2023)

Pretrained weights trained on badminton shuttlecock data but generalizes to
tennis balls (small, fast-moving objects on a court).
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import numpy.typing as npt
import torch
from collections import deque
from PIL import Image, ImageDraw

from src.models.tracknet import TrackNet, InpaintNet

# Constants matching TrackNetV3's training resolution
HEIGHT: int = 288
WIDTH: int = 512
SIGMA: float = 2.5
DELTA_T: float = 1 / math.sqrt(HEIGHT**2 + WIDTH**2)
COOR_TH: float = DELTA_T * 50


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


def load_tracknet(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TrackNet, int, str]:
    """Load TrackNet model from a checkpoint file.

    Args:
        checkpoint_path: Path to the TrackNet .pt checkpoint.
        device: Torch device to load the model onto.

    Returns:
        A tuple of (model, seq_len, bg_mode) where:
            - model: The loaded TrackNet in eval mode
            - seq_len: Sequence length the model expects (default 8)
            - bg_mode: Background mode ('concat', 'subtract', etc.)
    """
    ckpt: dict = torch.load(checkpoint_path, map_location=device)
    seq_len: int = ckpt["param_dict"]["seq_len"]
    bg_mode: str = ckpt["param_dict"]["bg_mode"]

    if bg_mode == "subtract":
        in_dim = seq_len
    elif bg_mode == "subtract_concat":
        in_dim = seq_len * 4
    elif bg_mode == "concat":
        in_dim = (seq_len + 1) * 3
    else:
        in_dim = seq_len * 3

    model: TrackNet = TrackNet(in_dim=in_dim, out_dim=seq_len)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, seq_len, bg_mode


def load_inpaintnet(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[InpaintNet, int]:
    """Load InpaintNet model from a checkpoint file.

    Args:
        checkpoint_path: Path to the InpaintNet .pt checkpoint.
        device: Torch device to load the model onto.

    Returns:
        A tuple of (model, seq_len) where:
            - model: The loaded InpaintNet in eval mode
            - seq_len: Sequence length for trajectory inpainting
    """
    ckpt: dict = torch.load(checkpoint_path, map_location=device)
    seq_len: int = ckpt["param_dict"]["seq_len"]
    model: InpaintNet = InpaintNet()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, seq_len


def generate_median_frame(
    video_path: str,
    max_samples: int = 1800,
) -> npt.NDArray[np.uint8]:
    """Compute the median frame from a video (used as background reference).

    Samples up to max_samples frames uniformly across the video and computes
    the per-pixel median, producing a static background estimate.

    Args:
        video_path: Path to the video file.
        max_samples: Maximum number of frames to sample.

    Returns:
        Median frame as RGB array with shape (3, HEIGHT, WIDTH), normalized
        and resized to model input dimensions.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_step: int = max(1, total_frames // max_samples)

    frames: list[npt.NDArray[np.uint8]] = []
    for i in range(0, total_frames, sample_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret: bool
        frame: npt.NDArray[np.uint8]
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Median across sampled frames, convert BGR -> RGB
    median: npt.NDArray[np.float64] = np.median(frames, axis=0)[..., ::-1]
    # Resize to model input size
    median_img: Image.Image = Image.fromarray(median.astype("uint8"))
    median_resized: npt.NDArray[np.uint8] = np.array(
        median_img.resize((WIDTH, HEIGHT))
    )
    # HWC -> CHW
    median_chw: npt.NDArray[np.uint8] = np.moveaxis(median_resized, -1, 0)
    return median_chw


def preprocess_sequence(
    frames: list[npt.NDArray[np.uint8]],
    median: npt.NDArray[np.uint8],
    bg_mode: str,
) -> npt.NDArray[np.float64]:
    """Preprocess a sequence of frames for TrackNet input.

    Resizes frames to (HEIGHT, WIDTH), converts to channel-first format,
    concatenates with the median background if bg_mode='concat', and
    normalizes to [0, 1].

    Args:
        frames: List of BGR frames (original resolution).
        median: Median background in CHW format, shape (3, HEIGHT, WIDTH).
        bg_mode: Background mode from the checkpoint ('concat', etc.).

    Returns:
        Preprocessed tensor-ready array with shape (C, HEIGHT, WIDTH) where
        C depends on bg_mode (e.g., 27 for 'concat' with seq_len=8).
    """
    processed: npt.NDArray[np.float64] = np.array([]).reshape(0, HEIGHT, WIDTH)

    for frame in frames:
        # BGR -> RGB
        rgb: npt.NDArray[np.uint8] = frame[..., ::-1]
        img: Image.Image = Image.fromarray(rgb)
        resized: npt.NDArray[np.uint8] = np.array(img.resize((WIDTH, HEIGHT)))
        # HWC -> CHW
        chw: npt.NDArray[np.uint8] = np.moveaxis(resized, -1, 0)
        processed = np.concatenate((processed, chw), axis=0)

    if bg_mode == "concat":
        processed = np.concatenate((median, processed), axis=0)

    processed = processed / 255.0
    return processed


def predict_ball_from_heatmap(
    heatmap: npt.NDArray[np.uint8],
) -> tuple[int, int]:
    """Extract ball (x, y) coordinates from a single-frame heatmap.

    Thresholds the heatmap at 0.5, finds contours, and returns the center
    of the largest contour's bounding box.

    Args:
        heatmap: Heatmap array with shape (HEIGHT, WIDTH), values in [0, 255].

    Returns:
        (cx, cy) pixel coordinates in the heatmap's resolution.
        Returns (0, 0) if no ball detected.
    """
    if np.amax(heatmap) == 0:
        return 0, 0

    contours, _ = cv2.findContours(
        heatmap.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0, 0

    rects: list[tuple[int, int, int, int]] = [
        cv2.boundingRect(c) for c in contours
    ]
    largest: tuple[int, int, int, int] = max(rects, key=lambda r: r[2] * r[3])
    x, y, w, h = largest
    return x + w // 2, y + h // 2


def generate_inpaint_mask(
    pred_dict: dict[str, list],
    th_h: float = 30.0,
) -> list[int]:
    """Generate a binary mask indicating which frames need trajectory inpainting.

    Marks frames where the ball disappeared but was likely still in play
    (based on the y-coordinate threshold), so InpaintNet can fill the gaps.

    Args:
        pred_dict: Prediction dict with keys 'Y' and 'Visibility'.
        th_h: Height threshold in pixels; only inpaint if surrounding
            y-coordinates exceed this (ball is not near top edge).

    Returns:
        List of 0/1 values, same length as pred_dict['Y'].
    """
    y: npt.NDArray[np.int64] = np.array(pred_dict["Y"])
    vis: npt.NDArray[np.int64] = np.array(pred_dict["Visibility"])
    mask: npt.NDArray[np.int64] = np.zeros_like(y)

    i: int = 0
    while i < len(vis):
        # Find start of gap
        while i < len(vis) - 1 and vis[i] == 1:
            i += 1
        j: int = i
        # Find end of gap
        while j < len(vis) - 1 and vis[j] == 0:
            j += 1
        if j == i:
            break
        elif i == 0 and y[j] > th_h:
            mask[:j] = 1
        elif (i > 1 and y[i - 1] > th_h) and (j < len(vis) and y[j] > th_h):
            mask[i:j] = 1
        i = j

    return mask.tolist()


def get_ensemble_weight(
    seq_len: int,
    mode: str = "weight",
) -> torch.Tensor:
    """Compute temporal ensemble weights for overlapping predictions.

    In 'weight' mode, uses a triangular weighting that gives more importance
    to the center of the sequence. In 'average' mode, uses uniform weights.

    Args:
        seq_len: Length of the input sequence.
        mode: Either 'weight' (positional) or 'average' (uniform).

    Returns:
        A 1D tensor of shape (seq_len,) summing to 1.0.
    """
    if mode == "average":
        return torch.ones(seq_len) / seq_len

    weight: torch.Tensor = torch.ones(seq_len)
    for i in range(math.ceil(seq_len / 2)):
        weight[i] = i + 1
        weight[seq_len - i - 1] = i + 1
    return weight / weight.sum()


def track_balls(
    video_path: str,
    tracknet_path: str,
    inpaintnet_path: str | None = None,
    eval_mode: str = "weight",
) -> dict[str, list]:
    """Run full ball tracking pipeline on a video.

    Runs TrackNet on overlapping 8-frame sequences with temporal ensemble,
    then optionally refines with InpaintNet for gap filling.

    Args:
        video_path: Path to the input video file.
        tracknet_path: Path to the TrackNet checkpoint (.pt).
        inpaintnet_path: Optional path to the InpaintNet checkpoint.
        eval_mode: Ensemble mode — 'weight', 'average', or 'nonoverlap'.

    Returns:
        Dict with keys 'Frame', 'X', 'Y', 'Visibility', each a list
        of per-frame values in the original video resolution.
    """
    device: torch.device = get_device()
    print(f"Using device: {device}")

    # Load TrackNet
    tracknet: TrackNet
    seq_len: int
    bg_mode: str
    tracknet, seq_len, bg_mode = load_tracknet(tracknet_path, device)
    print(f"TrackNet loaded (seq_len={seq_len}, bg_mode={bg_mode})")

    # Generate median background
    print("Computing median background frame...")
    median: npt.NDArray[np.uint8] = generate_median_frame(video_path)

    # Read all frames
    print("Reading video frames...")
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_scaler: float = w / WIDTH
    h_scaler: float = h / HEIGHT

    all_frames: list[npt.NDArray[np.uint8]] = []
    while True:
        ret: bool
        frame: npt.NDArray[np.uint8]
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()

    video_len: int = len(all_frames)
    print(f"Video: {video_len} frames, {w}x{h}")

    # --- TrackNet inference with temporal ensemble ---
    pred_dict: dict[str, list] = {
        "Frame": [],
        "X": [],
        "Y": [],
        "Visibility": [],
    }

    if eval_mode == "nonoverlap":
        step: int = seq_len
    else:
        step = 1

    num_sequences: int = max(1, video_len - seq_len + 1)
    weight: torch.Tensor = get_ensemble_weight(seq_len, eval_mode)

    if eval_mode != "nonoverlap":
        # Ensemble buffers
        buffer_size: int = seq_len - 1
        batch_i: torch.Tensor = torch.arange(seq_len)
        frame_i: torch.Tensor = torch.arange(seq_len - 1, -1, -1)
        y_pred_buffer: torch.Tensor = torch.zeros(
            (buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32
        )
        sample_count: int = 0

        for s in range(0, num_sequences, 1):
            # Get frame sequence
            end_idx: int = min(s + seq_len, video_len)
            frame_seq: list[npt.NDArray[np.uint8]] = all_frames[s:end_idx]
            # Pad if needed
            while len(frame_seq) < seq_len:
                frame_seq.append(frame_seq[-1])

            x: npt.NDArray[np.float64] = preprocess_sequence(
                frame_seq, median, bg_mode
            )
            x_tensor: torch.Tensor = (
                torch.from_numpy(x).float().unsqueeze(0).to(device)
            )

            with torch.no_grad():
                y_pred: torch.Tensor = tracknet(x_tensor).detach().cpu()

            # Ensemble accumulation
            y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)

            if sample_count < buffer_size:
                ensembled: torch.Tensor = (
                    y_pred_buffer[batch_i + 0, frame_i].sum(0) / (sample_count + 1)
                )
            else:
                ensembled = (
                    y_pred_buffer[batch_i + 0, frame_i] * weight[:, None, None]
                ).sum(0)

            # Extract ball position from ensembled heatmap
            heatmap: npt.NDArray[np.uint8] = (
                (ensembled.numpy() > 0.5).astype("uint8") * 255
            )
            cx, cy = predict_ball_from_heatmap(heatmap)
            cx_orig: int = int(cx * w_scaler)
            cy_orig: int = int(cy * h_scaler)
            vis: int = 0 if cx_orig == 0 and cy_orig == 0 else 1

            pred_dict["Frame"].append(s)
            pred_dict["X"].append(cx_orig)
            pred_dict["Y"].append(cy_orig)
            pred_dict["Visibility"].append(vis)

            sample_count += 1

            # Handle last frames
            if sample_count == num_sequences:
                y_zero_pad: torch.Tensor = torch.zeros(
                    (buffer_size, seq_len, HEIGHT, WIDTH), dtype=torch.float32
                )
                y_pred_buffer = torch.cat((y_pred_buffer, y_zero_pad), dim=0)
                for f in range(1, seq_len):
                    if s + f >= video_len:
                        break
                    ensembled = (
                        y_pred_buffer[batch_i + f, frame_i].sum(0) / (seq_len - f)
                    )
                    heatmap = (ensembled.numpy() > 0.5).astype("uint8") * 255
                    cx, cy = predict_ball_from_heatmap(heatmap)
                    cx_orig = int(cx * w_scaler)
                    cy_orig = int(cy * h_scaler)
                    vis = 0 if cx_orig == 0 and cy_orig == 0 else 1

                    pred_dict["Frame"].append(s + f)
                    pred_dict["X"].append(cx_orig)
                    pred_dict["Y"].append(cy_orig)
                    pred_dict["Visibility"].append(vis)

            # Trim buffer
            y_pred_buffer = y_pred_buffer[-buffer_size:]

            if (s + 1) % 20 == 0:
                print(f"  TrackNet: {s + 1}/{num_sequences} sequences...")

    else:
        # Non-overlap mode (simpler, faster)
        for s in range(0, video_len, seq_len):
            end_idx = min(s + seq_len, video_len)
            frame_seq = all_frames[s:end_idx]
            actual_len: int = len(frame_seq)
            while len(frame_seq) < seq_len:
                frame_seq.append(frame_seq[-1])

            x = preprocess_sequence(frame_seq, median, bg_mode)
            x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = tracknet(x_tensor).detach().cpu()

            y_np: npt.NDArray[np.float64] = y_pred.squeeze(0).numpy()
            for f in range(actual_len):
                heatmap = (y_np[f] > 0.5).astype("uint8") * 255
                cx, cy = predict_ball_from_heatmap(heatmap)
                cx_orig = int(cx * w_scaler)
                cy_orig = int(cy * h_scaler)
                vis = 0 if cx_orig == 0 and cy_orig == 0 else 1

                pred_dict["Frame"].append(s + f)
                pred_dict["X"].append(cx_orig)
                pred_dict["Y"].append(cy_orig)
                pred_dict["Visibility"].append(vis)

            print(f"  TrackNet: frames {s}-{end_idx}/{video_len}")

    print(f"TrackNet done: {len(pred_dict['Frame'])} frames, "
          f"{sum(pred_dict['Visibility'])} detections")

    # --- InpaintNet (trajectory gap filling) ---
    if inpaintnet_path:
        inpaintnet: InpaintNet
        inp_seq_len: int
        inpaintnet, inp_seq_len = load_inpaintnet(inpaintnet_path, device)
        print(f"InpaintNet loaded (seq_len={inp_seq_len})")

        inpaint_mask: list[int] = generate_inpaint_mask(
            pred_dict, th_h=h * 0.05
        )
        n_inpaint: int = sum(inpaint_mask)
        print(f"Frames to inpaint: {n_inpaint}")

        if n_inpaint > 0:
            # Prepare coordinate sequences for InpaintNet
            xs: npt.NDArray[np.float64] = (
                np.array(pred_dict["X"]) / (w_scaler * WIDTH)
            )
            ys: npt.NDArray[np.float64] = (
                np.array(pred_dict["Y"]) / (h_scaler * HEIGHT)
            )

            num_frames: int = len(xs)
            # Process in sliding windows
            for start in range(0, max(1, num_frames - inp_seq_len + 1)):
                end: int = min(start + inp_seq_len, num_frames)
                sl: slice = slice(start, end)

                coords: npt.NDArray[np.float64] = np.stack(
                    [xs[sl], ys[sl]], axis=-1
                )
                mask_seq: npt.NDArray[np.float64] = np.array(
                    inpaint_mask[sl]
                ).reshape(-1, 1)

                if mask_seq.sum() == 0:
                    continue

                # Pad if needed
                pad_len: int = inp_seq_len - len(coords)
                if pad_len > 0:
                    coords = np.pad(coords, ((0, pad_len), (0, 0)))
                    mask_seq = np.pad(mask_seq, ((0, pad_len), (0, 0)))

                coords_t: torch.Tensor = (
                    torch.from_numpy(coords).float().unsqueeze(0).to(device)
                )
                mask_t: torch.Tensor = (
                    torch.from_numpy(mask_seq).float().unsqueeze(0).to(device)
                )

                with torch.no_grad():
                    refined: torch.Tensor = (
                        inpaintnet(coords_t, mask_t).detach().cpu().squeeze(0)
                    )

                # Apply inpainted coordinates only where mask is 1
                for k in range(min(end - start, inp_seq_len)):
                    idx: int = start + k
                    if idx < num_frames and inpaint_mask[idx] == 1:
                        rx: float = refined[k, 0].item()
                        ry: float = refined[k, 1].item()
                        if rx < COOR_TH and ry < COOR_TH:
                            pred_dict["X"][idx] = 0
                            pred_dict["Y"][idx] = 0
                            pred_dict["Visibility"][idx] = 0
                        else:
                            pred_dict["X"][idx] = int(rx * WIDTH * w_scaler)
                            pred_dict["Y"][idx] = int(ry * HEIGHT * h_scaler)
                            pred_dict["Visibility"][idx] = 1

            print(f"InpaintNet done: {sum(pred_dict['Visibility'])} total detections")

    return pred_dict


def draw_ball_trajectory(
    frame: npt.NDArray[np.uint8],
    trajectory: deque[tuple[int, int] | None],
    radius: int = 3,
) -> npt.NDArray[np.uint8]:
    """Draw ball trajectory dots on a frame.

    Renders a trail of white circles with yellow outlines for recent
    ball positions.

    Args:
        frame: BGR frame to annotate.
        trajectory: Deque of (x, y) tuples or None for missed frames.
        radius: Circle radius in pixels.

    Returns:
        Annotated frame copy.
    """
    img: Image.Image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(img)

    for pos in trajectory:
        if pos is not None:
            x, y = pos
            bbox: tuple[int, int, int, int] = (
                x - radius, y - radius, x + radius, y + radius
            )
            draw.ellipse(bbox, fill="white", outline="yellow")

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def process_video(
    input_path: str,
    output_path: str,
    tracknet_path: str,
    inpaintnet_path: str | None = None,
    traj_len: int = 8,
) -> None:
    """Run ball tracking and produce an annotated output video.

    Combines TrackNet ball detection (+ optional InpaintNet gap filling)
    with trajectory visualization on the original video.

    Args:
        input_path: Path to the input video file.
        output_path: Path for the annotated output video.
        tracknet_path: Path to the TrackNet checkpoint.
        inpaintnet_path: Optional path to the InpaintNet checkpoint.
        traj_len: Number of past positions to draw as trajectory trail.
    """
    # Run tracking
    pred_dict: dict[str, list] = track_balls(
        input_path, tracknet_path, inpaintnet_path
    )

    # Write output video with trajectory overlay
    print("Writing output video...")
    cap: cv2.VideoCapture = cv2.VideoCapture(input_path)
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc: int = cv2.VideoWriter_fourcc(*"mp4v")
    out: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    trajectory: deque[tuple[int, int] | None] = deque(maxlen=traj_len)
    frame_idx: int = 0

    while True:
        ret: bool
        frame: npt.NDArray[np.uint8]
        ret, frame = cap.read()
        if not ret:
            break

        # Add current position to trajectory
        if frame_idx < len(pred_dict["Frame"]) and pred_dict["Visibility"][frame_idx]:
            trajectory.appendleft(
                (pred_dict["X"][frame_idx], pred_dict["Y"][frame_idx])
            )
        else:
            trajectory.appendleft(None)

        # Draw trajectory
        annotated: npt.NDArray[np.uint8] = draw_ball_trajectory(frame, trajectory)

        # Add info text
        vis_count: int = sum(pred_dict["Visibility"][:frame_idx + 1])
        info: str = (
            f"Frame {frame_idx} | "
            f"Ball: {'DETECTED' if frame_idx < len(pred_dict['Visibility']) and pred_dict['Visibility'][frame_idx] else 'MISSING'} | "
            f"Detections: {vis_count}/{frame_idx + 1}"
        )
        cv2.putText(
            annotated, info, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        out.write(annotated)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! {frame_idx} frames -> {output_path}")

    # Save CSV
    csv_path: str = output_path.replace(".mp4", "_ball.csv")
    import pandas as pd
    pd.DataFrame(pred_dict).to_csv(csv_path, index=False)
    print(f"Ball positions saved to {csv_path}")


if __name__ == "__main__":
    input_video: str = "data/raw/test_video/Test_Clip_1.mp4"
    output_video: str = "data/raw/test_video/Video_Output/ball_tracking.mp4"
    tracknet_ckpt: str = "models/TrackNet_best.pt"
    inpaintnet_ckpt: str = "models/InpaintNet_best.pt"

    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    print(f"Processing {input_video}...")
    process_video(input_video, output_video, tracknet_ckpt, inpaintnet_ckpt)
