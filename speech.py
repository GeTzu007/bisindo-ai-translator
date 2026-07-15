import threading

import pyttsx3


class TextToSpeech:
    """Text-to-speech yang aman dipanggil berulang kali."""

    def __init__(self, rate=165, volume=1.0):
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()
        self._is_speaking = False

    @property
    def is_speaking(self):
        return self._is_speaking

    def speak(self, text):
        text = text.strip()

        if not text or self._is_speaking:
            return

        thread = threading.Thread(
            target=self._speak_worker,
            args=(text,),
            daemon=True,
        )
        thread.start()

    def _speak_worker(self, text):
        with self._lock:
            self._is_speaking = True

            try:
                # Engine dibuat ulang setiap kali berbicara.
                # Ini lebih stabil untuk pemanggilan berulang.
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)

                engine.say(text)
                engine.runAndWait()
                engine.stop()

            except Exception as error:
                print(f"Gagal menjalankan suara: {error}")

            finally:
                self._is_speaking = False

    def stop(self):
        """Tidak diperlukan karena engine dibuat per pemanggilan."""
        pass