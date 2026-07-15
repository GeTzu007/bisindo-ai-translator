import json
from collections import Counter, deque
from pathlib import Path

import numpy as np

from config import (
    CLASS_NAMES_PATH,
    CONFIDENCE_THRESHOLD,
    MIN_STABLE_COUNT,
    MODEL_PATH,
    PREDICTION_HISTORY_SIZE,
)
from landmarks import create_hand_detector, detect_landmarks
from model import load_bisindo_model


class BisindoPredictor:
    """Menangani deteksi tangan dan prediksi gesture BISINDO."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        class_names_path: Path = CLASS_NAMES_PATH,
    ):
        self.class_names = self._load_class_names(
            class_names_path
        )

        self.model = load_bisindo_model(
            model_path=model_path
        )

        output_size = int(self.model.output_shape[-1])

        if output_size != len(self.class_names):
            raise ValueError(
                "Jumlah output model tidak sesuai dengan class_names.\n"
                f"Output model : {output_size}\n"
                f"Jumlah kelas: {len(self.class_names)}"
            )

        self.detector = create_hand_detector()

        self.prediction_history = deque(
            maxlen=PREDICTION_HISTORY_SIZE
        )

    @staticmethod
    def _load_class_names(
        class_names_path: Path,
    ) -> list[str]:
        """Memuat daftar kelas dari file JSON."""
        if not class_names_path.exists():
            raise FileNotFoundError(
                "File class_names.json tidak ditemukan:\n"
                f"{class_names_path}"
            )

        with open(
            class_names_path,
            "r",
            encoding="utf-8",
        ) as file:
            class_names = json.load(file)

        if not isinstance(class_names, list):
            raise ValueError(
                "class_names.json harus berisi list."
            )

        return class_names

    def _get_stable_label(self):
        """Mengambil prediksi paling dominan dari riwayat."""
        if len(self.prediction_history) < MIN_STABLE_COUNT:
            return None

        counter = Counter(self.prediction_history)
        label, count = counter.most_common(1)[0]

        if count < MIN_STABLE_COUNT:
            return None

        return label

    def reset_history(self):
        """Menghapus riwayat prediksi."""
        self.prediction_history.clear()

    def predict(self, frame) -> dict:
        """Memprediksi gesture dari satu frame kamera."""
        features, result = detect_landmarks(
            frame,
            self.detector,
        )

        if features is None:
            self.reset_history()

            return {
                "detected": False,
                "accepted": False,
                "label": None,
                "stable_label": None,
                "confidence": 0.0,
                "result": result,
            }

        model_input = np.expand_dims(
            features,
            axis=0,
        ).astype(np.float32)

        probabilities = self.model(
            model_input,
            training=False,
        ).numpy()[0]

        predicted_id = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[predicted_id]
        )

        label = self.class_names[predicted_id]

        accepted = (
            confidence >= CONFIDENCE_THRESHOLD
        )

        if accepted:
            self.prediction_history.append(label)
        else:
            self.reset_history()

        stable_label = self._get_stable_label()

        return {
            "detected": True,
            "accepted": accepted,
            "label": label,
            "stable_label": stable_label,
            "confidence": confidence,
            "result": result,
        }

    def close(self):
        """Menutup MediaPipe Hand Landmarker."""
        self.detector.close()