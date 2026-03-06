import speech_recognition as sr
import sys

def test_mic():
    r = sr.Recognizer()
    
    print("1. Checking Library Version...")
    print(f"✅ Library loaded. Version: {sr.__version__}")

    print("\n2. Searching for Microphones...")
    try:
        mic_list = sr.Microphone.list_microphone_names()
        if not mic_list:
            print("❌ No microphones found. Is your hardware plugged in?")
            return
        for i, name in enumerate(mic_list):
            print(f"   [{i}] {name}")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during mic search: {e}")
        return

    print("\n3. Attempting to Listen (3 seconds)...")
    try:
        with sr.Microphone() as source:
            print("   👂 Listening now... say something!")
            # Use a short timeout to prevent hanging
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            print("✅ Audio captured successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR during recording: {e}")
        print("\nTIP: This usually means PyAudio or PortAudio is broken.")

if __name__ == "__main__":
    test_mic()
