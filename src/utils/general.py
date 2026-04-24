from __future__ import annotations

import os
import cv2
import numpy as np
import numpy.typing as npt

from torch import nn

IMG_FORMAT = "png"


def show_model_size(model: nn.Module) -> None:
    """Estimate the size of the model.
    reference: https://discuss.pytorch.org/t/finding-model-size/130275/2

    Args:
        model (torch.nn.Module): target model
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.3f}MB")


def list_dirs(directory: str) -> list[str]:
    """Extension of os.listdir which return the directory pathes including input directory.

    Args:
        directory (str): Directory path

    Returns:
        (List[str]): Directory pathes with pathes including input directory
    """

    return sorted([os.path.join(directory, path) for path in os.listdir(directory)])


def to_img(image: npt.NDArray) -> npt.NDArray:
    """Convert the normalized image back to image format.

    Args:
        image (numpy.ndarray): Images with range in [0, 1]

    Returns:
        image (numpy.ndarray): Images with range in [0, 255]
    """

    image = image * 255
    image = image.astype("uint8")
    return image


def get_num_frames(rally_dir: str) -> int:
    """Return the number of frames in the video.

    Args:
        rally_dir (str): File path of the rally frame directory
            Format: '{data_dir}/{split}/match{match_id}/frame/{rally_id}'

    Returns:
        (int): Number of frames in the rally frame directory
    """

    try:
        frame_files = list_dirs(rally_dir)
    except Exception:
        raise ValueError(f"{rally_dir} does not exist.")
    frame_files = [f for f in frame_files if f.split(".")[-1] == IMG_FORMAT]
    return len(frame_files)


def get_rally_dirs(data_dir: str, split: str) -> list[str]:
    """Return all rally directories in the split.

    Args:
        data_dir (str): File path of the data root directory
        split (str): Split name

    Returns:
        rally_dirs: (List[str]): Rally directories in the split
            Format: ['{split}/match{match_id}/frame/{rally_id}', ...]
    """

    rally_dirs = []

    # Get all match directories in the split
    match_dirs = os.listdir(os.path.join(data_dir, split))
    match_dirs = [os.path.join(split, d) for d in match_dirs]
    match_dirs = sorted(match_dirs, key=lambda s: int(s.split("match")[-1]))

    # Get all rally directories in the match directory
    for match_dir in match_dirs:
        rally_dir = os.listdir(os.path.join(data_dir, match_dir, "frame"))
        rally_dir = sorted(rally_dir)
        rally_dir = [os.path.join(match_dir, "frame", d) for d in rally_dir]
        rally_dirs.extend(rally_dir)

    return rally_dirs


def generate_frames(video_file: str) -> list[npt.NDArray]:
    """Sample frames from the video.

    Args:
        video_file (str): File path of the video file

    Returns:
        frame_list (List[numpy.ndarray]): List of sampled frames
    """

    assert video_file[-4:] == ".mp4", "Invalid video file format."

    # Get camera parameters
    cap = cv2.VideoCapture(video_file)
    frame_list = []
    success = True

    # Sample frames until video end
    while success:
        success, frame = cap.read()
        if success:
            frame_list.append(frame)

    return frame_list
