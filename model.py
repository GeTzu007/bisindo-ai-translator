from pathlib import Path

import tensorflow as tf


def load_bisindo_model(
    model_path: Path,
) -> tf.keras.Model:
    """Memuat model BISINDO untuk inferensi."""

    if not model_path.exists():
        raise FileNotFoundError(
            "File model tidak ditemukan:\n"
            f"{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    return model