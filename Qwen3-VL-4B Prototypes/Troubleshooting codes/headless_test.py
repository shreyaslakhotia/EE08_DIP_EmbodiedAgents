import os
import sys
import pyttsx3
import speech_recognition as sr
from faster_whisper import WhisperModel
from picamera2 import Picamera2
import ollama

# 1. SETUP BRAINS (Faster-Whisper INT8)
print("Loading Faster-Whisper (Tiny.en / INT8 Mode)...")
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# 2. SETUP VOICE OUT (Linux ALSA/eSpeak)
print("Initializing Voice Output...")
try:
    engine = pyttsx3.init(driverName='espeak')
    engine.setProperty('rate', 160)
except Exception as e:
    print(f"Audio init warning: {e}")

def speak(text):
    engine.say(text)
    engine.runAndWait()

# 3. SETUP CAMERA (Picamera2)
print("Initializing Camera Sensor...")
picam2 = Picamera2()
# We don't need a continuous preview anymore, just configure it for taking stills
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# 4. SETUP MEMORY & MICROPHONE
history = [{'role': 'system', 'content': 'You are a helpful, concise Study Buddy.'}]
recognizer = sr.Recognizer()
# Aggressively cut out silence to speed up transcription
recognizer.pause_threshold = 0.5
recognizer.non_speaking_duration = 0.3
recognizer.dynamic_energy_threshold = True

print("\n" + "="*45)
print("🚀 HEADLESS STUDY BUDDY ACTIVE (CTRL+C to quit)")
print("="*45 + "\n")

try:
    while True:
        # --- 1. LISTEN ---
        with sr.Microphone() as source:
            print("\n[🎙️ LISTENING...]")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=8)
            except sr.WaitTimeoutError:
                continue

        # --- 2. TRANSCRIBE ---
        print("[⚙️ TRANSCRIBING...]")
        with open("temp.wav", "wb") as f: 
            f.write(audio.get_wav_data())
        
        segments, info = whisper_model.transcribe("temp.wav", beam_size=1)
        user_text = " ".join([segment.text for segment in segments]).strip()
        
        if not user_text or len(user_text) < 4:
            continue
            
        print(f"👤 YOU: {user_text}")

        # --- 3. VISION & EMOTION ROUTING ---
        images = []
        vision_keywords = ["look", "see", "show", "watch", "emotion", "feeling"]
        is_vision = any(word in user_text.lower() for word in vision_keywords)
        
        if is_vision:
            print("[📸 TAKING PHOTO...]")
            img_path = "vision_temp.jpg"
            # Instantly grab a frame only when requested
            picam2.capture_file(img_path) 
            images = [img_path]
            
            if "emotion" in user_text.lower() or "feeling" in user_text.lower():
                user_text += " (System Note: Please analyze my facial expressions and posture to tell me what emotion I am displaying.)"

        # --- 4. AI INFERENCE ---
        print("[🧠 THINKING...]")
        history.append({'role': 'user', 'content': user_text, 'images': images})
        
        print("🤖 BUDDY: ", end="", flush=True)
        full_reply = ""
        
        # Simple synchronous streaming loop (no complex asyncio needed)
        stream = ollama.chat(
            model='qwen3-vl:4b', 
            messages=history, 
            stream=True,
            keep_alive=-1
        )
        
        for chunk in stream:
            token = chunk['message']['content']
            # Print to terminal in real-time
            print(token, end="", flush=True)
            full_reply += token
            
        print() # Add a newline when the AI finishes
        
        # --- 5. SPEAK & CLEANUP ---
        speak(full_reply)
        history.append({'role': 'assistant', 'content': full_reply})
        
        # Strip the heavy image data from the log to prevent slowdowns on the next turn
        for msg in history:
            if 'images' in msg:
                del msg['images']

except KeyboardInterrupt:
    # Graceful exit when you press CTRL+C
    print("\n[🛑 SHUTTING DOWN...]")
    picam2.stop()
    sys.exit(0)