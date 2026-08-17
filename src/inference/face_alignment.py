"""Five-point face alignment for the SFace input."""

from __future__ import annotations

import cv2
import numpy as np


_SFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(
    image: np.ndarray,
    yunet_landmarks: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Warp YuNet's five landmarks into the 112x112 SFace coordinate system."""
    if len(yunet_landmarks) != 5:
        raise ValueError("Face alignment requires exactly five landmarks")

    # YuNet returns right eye, left eye, nose, right mouth, left mouth.
    source = np.asarray(
        [
            yunet_landmarks[1],
            yunet_landmarks[0],
            yunet_landmarks[2],
            yunet_landmarks[4],
            yunet_landmarks[3],
        ],
        dtype=np.float32,
    )
    transform, _ = cv2.estimateAffinePartial2D(source, _SFACE_TEMPLATE)
    if transform is None:
        raise ValueError("Could not estimate a face alignment transform")

    return cv2.warpAffine(image, transform, (112, 112))
