import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from picamera2 import Picamera2
from ollama import AsyncClient
from faster_whisper import WhisperModel
from gtts import gTTS
import speech_recognition as sr
import pyttsx3
import threading
import asyncio
import re
import os
import os

# ==========================================
# CONFIGURATION
# ==========================================
MAC_IP = "10.91.51.112"  # <-- CHANGE THIS TO YOUR MACBOOK'S IP
MODEL_NAME = "qwen3-vl:2b-instruct-q4_k_m"


# ==========================================
# 1. THE BRAIN (Remote Network Client)
# ==========================================
class RemoteBrain:
    """Handles all network communication with the MacBook server."""
    def __init__(self, server_ip, model):
        self.client = AsyncClient(host=f'http://{server_ip}:11434')
        self.model = model
        self.history = [{'role': 'system', 'content': 'You are an embodied Study Buddy. Provide concise, helpful answers.'}]

    async def generate_response_stream(self, user_text, image_path=None):
        """Yields tokens as they stream in from the Mac server."""
        images = [image_path] if image_path else []
        self.history.append({'role': 'user', 'content': user_text, 'images': images})
        
        full_reply = ""
        try:
            stream = await self.client.chat(
                model=self.model,
                messages=self.history,
                stream=True,
                keep_alive=-1
            )
            async for chunk in stream:
                token = chunk['message']['content']
                full_reply += token
                yield token

            self.history.append({'role': 'assistant', 'content': full_reply})
            self._clean_history()
        except Exception as e:
            yield f"\n[NETWORK ERROR: Cannot reach MacBook at {MAC_IP}. Details: {e}]"

    def _clean_history(self):
        """Removes heavy image payloads from past messages to prevent network lag."""
        for msg in self.history:
            if 'images' in msg:
                del msg['images']


# ==========================================
# 2. THE EYES (Hardware Camera)
# ==========================================
class VisionSystem:
    """Manages the Picamera2 hardware and frame captures."""
    def __init__(self):
        print("Initializing Vision System...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

    def get_current_frame(self):
        """Returns the current frame as a PIL Image."""
        try:
            frame = self.picam2.capture_array("main")
            return Image.fromarray(frame)
        except Exception:
            return None

    def save_frame(self, img_obj, filepath="vision_temp.jpg"):
        """Saves a PIL image to disk for the AI to read."""
        if img_obj:
            img_obj.save(filepath)
            return filepath
        return None

    def shutdown(self):
        self.picam2.stop()


# ==========================================
# 3. THE MOUTH & EARS (Audio Processing)
# ==========================================
class AudioSystem:
    """Manages text-to-speech and speech-to-text."""
    def __init__(self):
        print("Initializing Audio System...")
        # Speech-to-Text Setup
        self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5

    def listen(self, status_callback):
        """Listens to the microphone and transcribes audio to text."""
        with sr.Microphone() as source:
            status_callback("LISTENING...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)
                status_callback("TRANSCRIBING...")
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                segments, _ = self.whisper_model.transcribe("temp.wav", beam_size=1)
                text = " ".join([segment.text for segment in segments])
                return text.strip()
            except Exception as e:
                print(f"Voice Error: {e}")
                return None

    def _clean_for_speech(self, text):
        """Sanitizes LLM markdown and emojis so the TTS doesn't read them."""
        # 1. Strip out markdown symbols (asterisks, hashes, backticks, brackets)
        cleaned = re.sub(r'[\*\#\[\]\(\)\`\_]', '', text)
        
        # 2. Strip out emojis and completely foreign characters 
        # (Keeps only word characters, spaces, and basic punctuation)
        cleaned = re.sub(r'[^\w\s.,!?\'"\-:]', '', cleaned)
        
        # 3. Condense any accidental double-spaces left behind
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned

    def speak(self, text):
        """Cleans the text and speaks it out loud in a non-blocking thread."""
        def run_speech():
            cleaned_text = self._clean_for_speech(text)
            
            # Prevent the engine from crashing if the AI only replied with an emoji
            if not cleaned_text:
                return 
                
            try:
                # Generate natural voice (tld='com' is US English, 'co.uk' is British)
                tts = gTTS(text=cleaned_text, lang='en', tld='com')
                audio_file = "speech.mp3"
                tts.save(audio_file)
                
                # Play the file silently via the Linux terminal using mpg123
                os.system(f"mpg123 -q {audio_file}")
            except Exception as e:
                print(f"TTS Error: {e}")
                
        threading.Thread(target=run_speech, daemon=True).start()


# ==========================================
# 4. THE INTERFACE (Main Orchestrator)
# ==========================================
class StudyBuddyApp:
    """The main Tkinter GUI that glues all systems together."""
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy Edge Client")
        self.root.geometry("800x900")

        # Initialize Sub-systems
        self.brain = RemoteBrain(server_ip=MAC_IP, model=MODEL_NAME)
        self.vision = VisionSystem()
        self.audio = AudioSystem()

        # State Variables
        self.running = True
        self.processing = False
        self.current_frame_img = None

        self._build_ui()
        
        # Start background loops
        self.update_video_feed()
        threading.Thread(target=self.voice_loop, daemon=True).start()

    def _build_ui(self):
        self.status = tk.Label(self.root, text="STATUS: READY", font=("Arial", 12, "bold"), fg="blue")
        self.status.pack(pady=5)

        self.vid_label = tk.Label(self.root, bg="black")
        self.vid_label.pack(pady=5)

        self.chat_log = scrolledtext.ScrolledText(self.root, width=70, height=18)
        self.chat_log.pack(pady=10)

        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=5, padx=10, fill='x')
        
        self.user_entry = tk.Entry(self.input_frame, font=("Arial", 14))
        self.user_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.user_entry.bind("<Return>", lambda event: self.handle_input())

        self.send_btn = tk.Button(self.input_frame, text="SEND", command=self.handle_input)
        self.send_btn.pack(side='right')

        self.exit_btn = tk.Button(self.root, text="SHUTDOWN", command=self.shutdown, bg="red", fg="white")
        self.exit_btn.pack(pady=10)

    def update_video_feed(self):
        """Updates the GUI video feed only when the system isn't busy."""
        if self.running:
            if not self.processing:
                img = self.vision.get_current_frame()
                if img:
                    self.current_frame_img = img.copy()
                    img_gui = img.resize((400, 300))
                    imgtk = ImageTk.PhotoImage(image=img_gui)
                    self.vid_label.imgtk = imgtk
                    self.vid_label.configure(image=imgtk)
            
            self.root.after(150, self.update_video_feed)

    def set_status(self, msg):
        self.status.config(text=f"STATUS: {msg}")

    def voice_loop(self):
        """Continuously listens for voice input in the background."""
        while self.running:
            if not self.processing:
                user_text = self.audio.listen(self.set_status)
                if user_text and len(user_text) > 4:
                    self.trigger_ai_interaction(user_text)

    def handle_input(self):
        """Triggered by typing in the text box."""
        text = self.user_entry.get().strip()
        if text and not self.processing:
            self.user_entry.delete(0, tk.END)
            self.trigger_ai_interaction(text)

    def trigger_ai_interaction(self, text):
        """Bridges the sync Tkinter loop to the async AI network call."""
        self.processing = True
        self.chat_log.insert(tk.END, f"You: {text}\n\n")
        self.chat_log.see(tk.END)

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.process_ai_stream(text))
            loop.close()
            
        threading.Thread(target=run_async, daemon=True).start()

    async def process_ai_stream(self, text):
        """Handles vision keywords and streams the network response."""
        vision_keywords = ["look", "see", "show", "analyze", "watch"]
        image_path = None
        
        if any(word in text.lower() for word in vision_keywords):
            self.set_status("📸 TRANSMITTING PHOTO...")
            image_path = self.vision.save_frame(self.current_frame_img)

        self.set_status("📡 AWAITING MACBOOK...")
        self.chat_log.insert(tk.END, "Agent: ")
        
        full_reply = ""
        # Stream the tokens from the RemoteBrain
        async for token in self.brain.generate_response_stream(text, image_path):
            full_reply += token
            self.chat_log.insert(tk.END, token)
            self.chat_log.see(tk.END)

        self.chat_log.insert(tk.END, "\n\n")
        self.audio.speak(full_reply)
        
        self.processing = False
        self.set_status("READY")

    def shutdown(self):
        self.running = False
        self.vision.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StudyBuddyApp(root)
    root.mainloop()