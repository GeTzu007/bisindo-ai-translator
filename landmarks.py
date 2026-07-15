from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import (
    HAND_LANDMARKER_PATH,
    MAX_NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    NUM_HAND_FEATURES,
)


def create_hand_detector(
    model_path: Path = HAND_LANDMARKER_PATH,
):
    """Membuat MediaPipe Hand Landmarker."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model MediaPipe tidak ditemukan:\n{model_path}"
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path),
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=(
            MIN_HAND_DETECTION_CONFIDENCE
        ),
        min_hand_presence_confidence=(
            MIN_HAND_PRESENCE_CONFIDENCE
        ),
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


def normalize_hand_landmarks(
    hand_landmarks,
) -> np.ndarray:
    """
    Menormalisasi 21 landmark terhadap pergelangan tangan
    dan ukuran tangan.

    Output:
        63 fitur: 21 landmark × koordinat x, y, z.
    """
    coordinates = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks
        ],
        dtype=np.float32,
    )

  
    coordinates -= coordinates[0]


    scale = np.max(
        np.linalg.norm(
            coordinates[:, :2],
            axis=1,
        )
    )

    if scale > 0:
        coordinates /= scale

    return coordinates.flatten()


def extract_two_hand_features(result):
    """
    Menghasilkan 126 fitur:

    - 63 fitur tangan kiri
    - 63 fitur tangan kanan

    Bagian yang tidak terdeteksi akan diisi nol.
    """
    if not result.hand_landmarks:
        return None

    left_features = np.zeros(
        NUM_HAND_FEATURES,
        dtype=np.float32,
    )

    right_features = np.zeros(
        NUM_HAND_FEATURES,
        dtype=np.float32,
    )

    for index, hand_landmarks in enumerate(
        result.hand_landmarks
    ):
        features = normalize_hand_landmarks(
            hand_landmarks
        )

        handedness = (
            result.handedness[index][0]
            .category_name
            .lower()
        )

        if handedness == "left":
            left_features = features

        elif handedness == "right":
            right_features = features

    return np.concatenate(
        [left_features, right_features]
    )


def detect_landmarks(frame, detector):
    """
    Mendeteksi tangan dari frame OpenCV dan menghasilkan
    fitur landmark untuk model Neural Network.
    """
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    rgb_frame = np.ascontiguousarray(rgb_frame)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = detector.detect(mp_image)

    features = extract_two_hand_features(result)

    return features, result