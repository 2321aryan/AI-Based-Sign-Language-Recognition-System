import pyttsx3
import threading

class AudioOutput:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Optional: Configure voice properties here if needed
        # voices = self.engine.getProperty('voices')
        # self.engine.setProperty('voice', voices[0].id) 

    def speak(self, text):
        """
        Speaks the given text in a separate thread to avoid blocking the main loop.
        """
        threading.Thread(target=self._speak_thread, args=(text,), daemon=True).start()

    def _speak_thread(self, text):
        try:
            # Re-initialize engine inside thread to avoid loop conflicts
            # This is safer for threaded usage than sharing a global engine
            engine = pyttsx3.init()
            engine.say(text)
            if engine._inLoop:
                engine.endLoop()
            engine.runAndWait()
        except Exception as e:
            # Common error if loop is somehow stuck, just ignore to keep app running
            print(f"Audio Logic Error (Non-fatal): {e}")
