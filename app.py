import time

import cv2


from speech import TextToSpeech

from config import (
    AUTO_SPEAK_DELAY_SECONDS,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    COLOR_ACCENT,
    COLOR_BACKGROUND,
    COLOR_ERROR,
    COLOR_PANEL,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_WARNING,
    ESC_KEY,
    MAX_SENTENCE_LENGTH,
    RELEASE_FRAME_COUNT,
    WINDOW_NAME,
)
from predictor import BisindoPredictor


def create_camera():
    """Membuka webcam."""
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW,
    )

    if not camera.isOpened():
        raise RuntimeError(
            "Kamera tidak dapat dibuka. "
            "Pastikan kamera tidak digunakan aplikasi lain."
        )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    return camera



class SentenceBuilder:
    """Menyusun hasil gesture menjadi teks."""

    def __init__(self):
        self.items = []

    @staticmethod
    def is_letter(label: str) -> bool:
        """Memeriksa apakah label adalah huruf A-Z."""
        return (
            len(label) == 1
            and label.isalpha()
        )

    def add(self, label: str):
        """Menambahkan hasil gesture ke kalimat."""
        if not label:
            return

        current_text = self.get_text()

        if len(current_text) >= MAX_SENTENCE_LENGTH:
            return

        self.items.append(label)

    def delete_last(self):
        """Menghapus hasil gesture terakhir."""
        if self.items:
            self.items.pop()

    def clear(self):
        """Menghapus seluruh hasil kalimat."""
        self.items.clear()

    def get_text(self) -> str:
        """
        Menyusun token menjadi teks.

        Huruf tunggal akan disambungkan.
        Label kata akan dipisahkan dengan spasi.
        """
        final_parts = []
        letter_buffer = ""

        for item in self.items:
            if self.is_letter(item):
                letter_buffer += item.upper()
                continue

            if letter_buffer:
                final_parts.append(letter_buffer)
                letter_buffer = ""

            final_parts.append(item)

        if letter_buffer:
            final_parts.append(letter_buffer)

        return " ".join(final_parts)




class GestureLock:
    """
    Mencegah gesture yang sama ditambahkan berkali-kali.

    Kunci dibuka ketika:
    - tangan tidak terdeteksi beberapa frame; atau
    - gesture stabil berubah menjadi label lain.
    """

    def __init__(self):
        self.locked_label = None
        self.release_counter = 0

    def update(
        self,
        stable_label,
        hand_detected: bool,
    ):
        """
        Mengembalikan label yang boleh ditambahkan.

        Return:
            str atau None
        """
        if not hand_detected:
            self.release_counter += 1

            if self.release_counter >= RELEASE_FRAME_COUNT:
                self.reset()

            return None

        self.release_counter = 0

        if stable_label is None:
            return None

        
        if self.locked_label is None:
            self.locked_label = stable_label
            return stable_label

        
        if stable_label != self.locked_label:
            self.locked_label = stable_label
            return stable_label

        
        return None

    def reset(self):
        """Membuka kunci gesture."""
        self.locked_label = None
        self.release_counter = 0


def draw_filled_rectangle(
    frame,
    top_left,
    bottom_right,
    color,
    opacity=0.75,
):
    """Menggambar panel transparan."""
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        top_left,
        bottom_right,
        color,
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        opacity,
        frame,
        1 - opacity,
        0,
        frame,
    )


def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    color=COLOR_TEXT,
    thickness=2,
):
    """Menggambar teks pada frame."""
    cv2.putText(
        img=frame,
        text=text,
        org=position,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=scale,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def shorten_text(text: str, max_length=60) -> str:
    """Memotong teks agar tidak keluar dari layar."""
    if len(text) <= max_length:
        return text

    return "..." + text[-max_length:]


def draw_interface(
    frame,
    prediction,
    sentence_text,
    fps,
    gesture_locked,
):
    """Menggambar tampilan aplikasi."""
    height, width = frame.shape[:2]

    # Panel bagian atas
    draw_filled_rectangle(
        frame,
        (0, 0),
        (width, 150),
        COLOR_BACKGROUND,
        opacity=0.82,
    )

    # Panel bagian bawah
    draw_filled_rectangle(
        frame,
        (0, height - 140),
        (width, height),
        COLOR_PANEL,
        opacity=0.85,
    )

    draw_text(
        frame,
        "BISINDO AI TRANSLATOR",
        (30, 40),
        scale=0.9,
        color=COLOR_ACCENT,
        thickness=2,
    )

    confidence = prediction["confidence"]

    if not prediction["detected"]:
        status = "Tangan tidak terdeteksi"
        status_color = COLOR_ERROR

    elif not prediction["accepted"]:
        status = (
            f"Confidence rendah: "
            f"{confidence * 100:.1f}%"
        )
        status_color = COLOR_WARNING

    elif prediction["stable_label"] is None:
        status = (
            f"Mendeteksi: {prediction['label']} "
            f"({confidence * 100:.1f}%)"
        )
        status_color = COLOR_WARNING

    else:
        status = (
            f"Prediksi: {prediction['stable_label']} "
            f"({confidence * 100:.1f}%)"
        )
        status_color = COLOR_SUCCESS

    draw_text(
        frame,
        status,
        (30, 85),
        scale=0.8,
        color=status_color,
    )

    lock_status = (
        f"Gesture Lock: {gesture_locked}"
        if gesture_locked
        else "Gesture Lock: terbuka"
    )

    draw_text(
        frame,
        lock_status,
        (30, 125),
        scale=0.55,
        color=COLOR_TEXT,
        thickness=1,
    )

    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        (width - 150, 40),
        scale=0.6,
        color=COLOR_TEXT,
        thickness=1,
    )

    draw_text(
        frame,
        "HASIL TERJEMAHAN",
        (30, height - 95),
        scale=0.6,
        color=COLOR_ACCENT,
        thickness=2,
    )

    display_sentence = shorten_text(sentence_text)

    if not display_sentence:
        display_sentence = "-"

    draw_text(
        frame,
        display_sentence,
        (30, height - 50),
        scale=0.85,
        color=COLOR_TEXT,
        thickness=2,
    )

    draw_text(
        frame,
        "S: suara | BACKSPACE: hapus | C: bersihkan | ESC: keluar",
        (width - 460, height - 15),
        scale=0.45,
        color=COLOR_TEXT,
        thickness=1,
    )




def main():
    """Menjalankan aplikasi BISINDO Translator."""
    print("Memuat model BISINDO...")

    predictor = BisindoPredictor()
    sentence_builder = SentenceBuilder()
    gesture_lock = GestureLock()
    speech = TextToSpeech()
    camera = create_camera()

    previous_time = time.perf_counter()
    fps = 0.0
    last_input_time = None
    last_spoken_text = ""

    print("Model berhasil dimuat.")
    print("Gesture stabil akan masuk otomatis.")
    print("Turunkan tangan untuk mengulang gesture yang sama.")
    print("Tekan BACKSPACE untuk menghapus.")
    print("Tekan C untuk membersihkan.")
    print("Tekan ESC untuk keluar.")

    try:
        while True:
            success, original_frame = camera.read()

            if not success:
                print("Gagal membaca frame kamera.")
                break

            prediction = predictor.predict(original_frame)

            stable_label = prediction["stable_label"]

            label_to_add = gesture_lock.update(
                stable_label=stable_label,
                hand_detected=prediction["detected"],
            )

            if label_to_add is not None:
                sentence_builder.add(label_to_add)
                last_input_time = time.time()

                print(f"Ditambahkan: {label_to_add}")

                predictor.reset_history()
            sentence_text = sentence_builder.get_text()

            if (
                last_input_time is not None
                and sentence_text
                and sentence_text != last_spoken_text
            ):
                idle_time = time.time() - last_input_time

                if idle_time >= AUTO_SPEAK_DELAY_SECONDS:
                    speech.speak(sentence_text)

                    last_spoken_text = sentence_text
                    last_input_time = None


            current_time = time.perf_counter()
            elapsed_time = current_time - previous_time

            if elapsed_time > 0:
                current_fps = 1.0 / elapsed_time

                fps = (
                    0.90 * fps + 0.10 * current_fps
                    if fps > 0
                    else current_fps
                )

            previous_time = current_time

            

            display_frame = cv2.flip(
                original_frame,
                1,
            )

            draw_interface(
                frame=display_frame,
                prediction=prediction,
                sentence_text=sentence_text,
                fps=fps,
                gesture_locked=gesture_lock.locked_label,
            )

            cv2.imshow(
                WINDOW_NAME,
                display_frame,
            )

           

            key = cv2.waitKey(1) & 0xFF

            if key == ESC_KEY:
                break

            elif key == ord("c"):
                sentence_builder.clear()
                gesture_lock.reset()
                predictor.reset_history()

                last_input_time = None
                last_spoken_text = ""

            elif key in (8, 127):
                sentence_builder.delete_last()
                gesture_lock.reset()
                predictor.reset_history()

                last_input_time = time.time()
                last_spoken_text = ""

            elif key == ord("s"):
                sentence_text = sentence_builder.get_text()

                if sentence_text:
                    speech.speak(sentence_text)

            elif key ==ord("f"):
                fullscreen = not fullscreen

                if fullscreen:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                else:
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL,
                    )
                


    finally:
        speech.stop()
        camera.release()
        predictor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()