import whisper
import numpy as np
import time

def test_whisper():
    try:
        print("1. Initializing Whisper (Tiny Model)...")
        # 'tiny' is the least demanding model for a Pi 4
        model = whisper.load_model("tiny")
        print("✅ Model loaded successfully.")

        print("2. Creating dummy audio data...")
        # Create 5 seconds of silent audio (16kHz mono)
        audio = np.zeros(16000 * 5, dtype=np.float32)

        print("3. Attempting transcription...")
        start_time = time.time()
        
        # 'fp16=False' is CRITICAL for Raspberry Pi (it has no half-precision hardware)
        result = model.transcribe(audio, fp16=False)
        
        duration = time.time() - start_time
        print(f"✅ Transcription complete in {duration:.2f} seconds.")
        print(f"Resulting Text: '{result['text']}'")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_whisper()
