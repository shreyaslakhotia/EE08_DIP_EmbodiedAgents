import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from picamera2 import Picamera2
from faster_whisper import WhisperModel
import speech_recognition as sr
from gtts import gTTS
import threading
import requests
import base64
import json
import time
import os
import re
from face_controller import FaceController

# ==========================================
# CONFIGURATION
# ==========================================
MAC_HOSTNAME = "Ongs-MacBook-Pro.local"
MAC_IP = MAC_HOSTNAME
MODEL_NAME = "stanky2"
TELEGRAM_REDIRECT_TEXT = (
    "this would be easier to explain properly on the Telegram interface where I can "
    "format things clearly. send it there and I'll walk you through it step by step."
)


# ==========================================
# 1. THE BRAIN (Raw HTTP REST Client)
# ==========================================
class RemoteBrain:
    """Handles all network communication with the MacBook server via pure HTTP."""
    def __init__(self, server_ip, model):
        self.server_url = f"http://{server_ip}:11434/api/chat"
        self.model = model
        self.history = [{'role': 'system', 'content': 'You are an embodied Study Buddy. Provide concise, helpful answers.'}]

    def generate_response_stream(self, user_text, image_path=None):
        """Yields tokens with built-in retries for robotic resilience."""
        images_b64 = []
        if image_path:
            try:
                with open(image_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                    images_b64.append(b64_string)
            except Exception as e:
                print(f"Image Encoding Error: {e}")

        # 1. Append the user message to history ONCE before the retry loop
        message = {'role': 'user', 'content': user_text}
        if images_b64:
            message['images'] = images_b64
        self.history.append(message)
        
        payload = {
            "model": self.model,
            "messages": self.history,
            "stream": True,
            "keep_alive": -1
        }
        
        max_retries = 3
        full_reply = ""

        # 2. Start the Retry Loop
        for attempt in range(max_retries):
            try:
                # Added a timeout (10s to connect, 30s for the first token)
                with requests.post(self.server_url, json=payload, stream=True, timeout=(10, 30)) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                token = chunk["message"]["content"]
                                full_reply += token
                                yield token

                # 3. SUCCESS: If we get here, the stream finished. Update history and EXIT.
                self.history.append({'role': 'assistant', 'content': full_reply})
                self._clean_history()
                return 

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                # 4. FAILURE: If it's not the last attempt, wait and try again
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1) # Exponential-ish backoff
                    yield f"\n[Signal weak... retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})]"
                    time.sleep(wait_time)
                else:
                    # Final failure
                    yield f"\n[OFFLINE: TARS cannot reach the MacBook. Check the server at {MAC_IP}.]"
                    # Remove the failed message from history so the context stays clean
                    if self.history and self.history[-1]['role'] == 'user':
                        self.history.pop()

    def _clean_history(self):
        """Removes heavy image payloads from past messages to prevent network lag."""
        for msg in self.history:
            if 'images' in msg:
                del msg['images']

    def silent_observe(self, image_path):
        """A stateless API call that checks the user's emotion without saving to chat history."""
        try:
            with open(image_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')
            
            # The strict prompt that forces the AI to evaluate your state
            observation_prompt = (
                "You are an empathetic study buddy observing the user through a camera. "
                "Analyze their facial expression and body language. "
                "If they look visibly frustrated, confused, tired, or have their head in their hands, "
                "generate a very brief, friendly, supportive interruption (e.g., 'You look a bit stuck, want to bounce some ideas off me?' or 'Remember to take a breath!'). "
                "If they look focused, neutral, or are just reading normally, YOU MUST OUTPUT EXACTLY AND ONLY THE WORD: SILENCE."
            )

            payload = {
                "model": self.model,
                "messages": [{'role': 'user', 'content': observation_prompt, 'images': [b64_string]}],
                "stream": False # We don't need to stream this, just get the final verdict
            }
            
            response = requests.post(self.server_url, json=payload, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            return data.get("message", {}).get("content", "SILENCE").strip()
            
        except Exception as e:
            print(f"Observation Error: {e}")
            return "SILENCE"


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
        try:
            # Grab the raw array from the camera
            frame = self.picam2.capture_array("main")
            
            # THE FIX: Reverse the color channels from BGR to RGB
            # This uses standard Numpy slicing to flip the 3rd dimension (colors)
            frame_rgb = frame[:, :, ::-1] 
            
            return Image.fromarray(frame_rgb)
        except Exception:
            return None

    def save_frame(self, img_obj, filepath="vision_temp.jpg"):
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
        self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5

    def listen(self, status_callback):
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
        cleaned = re.sub(r'[\*\#\[\]\(\)\`\_]', '', text)
        cleaned = re.sub(r'[^\w\s.,!?\'"\-:]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def speak(self, text, on_start=None, on_end=None):
        def run_speech():
            cleaned_text = self._clean_for_speech(text)
            if not cleaned_text:
                if callable(on_end):
                    on_end()
                return 
            try:
                if callable(on_start):
                    on_start()
                tts = gTTS(text=cleaned_text, lang='en', tld='com')
                audio_file = "speech.mp3"
                tts.save(audio_file)
                os.system(f"mpg123 -q {audio_file}")
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                if callable(on_end):
                    on_end()
                
        speech_thread = threading.Thread(target=run_speech, daemon=True)
        speech_thread.start()
        return speech_thread


# ==========================================
# 4. THE INTERFACE (Main Orchestrator)
# ==========================================
class StudyBuddyApp:
    """The main Tkinter GUI that glues all systems together synchronously."""
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy Edge Client")
        self.root.geometry("800x900")

        self.brain = RemoteBrain(server_ip=MAC_IP, model=MODEL_NAME)
        self.vision = VisionSystem()
        self.audio = AudioSystem()
        self.face = FaceController(
            width=480,
            height=320,
            debug_output_dir="face_debug",
            save_debug_frames=True,
        )
        self.face.set_idle()

        self.running = True
        self.processing = False
        self.current_frame_img = None

        self._build_ui()
        
        self.update_video_feed()
        threading.Thread(target=self.voice_loop, daemon=True).start()

        # --- NEW: Start the proactive observer ---
        threading.Thread(target=self.proactive_heartbeat_loop, daemon=True).start()

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
        while self.running:
            if not self.processing:
                user_text = self.audio.listen(self.set_status)
                if user_text and len(user_text) > 4:
                    self.trigger_ai_interaction(user_text)

    def handle_input(self):
        text = self.user_entry.get().strip()
        if text and not self.processing:
            self.user_entry.delete(0, tk.END)
            self.trigger_ai_interaction(text)

    def trigger_ai_interaction(self, text):
        """Bridges the Tkinter UI to the background network thread. NO ASYNC."""
        self.processing = True
        self.chat_log.insert(tk.END, f"You: {text}\n\n")
        self.chat_log.see(tk.END)

        # Start a standard thread. No asyncio loops at all.
        threading.Thread(target=self.process_ai_stream, args=(text,), daemon=True).start()

    def _should_redirect_to_telegram(self, text):
        lower = (text or "").lower()
        redirect_keywords = [
            "code",
            "equation",
            "derive",
            "proof",
            "formula",
            "step by step",
            "debug",
            "algorithm",
            "syntax",
            "explain in detail",
        ]
        return any(keyword in lower for keyword in redirect_keywords)

    def _tts_finish_face_state(self):
        self.face.stop_talking()
        self.face.set_idle()

    def process_ai_stream(self, text):
        """Handles vision routing and processes the network stream synchronously."""
        if self._should_redirect_to_telegram(text):
            self.chat_log.insert(tk.END, f"Agent: {TELEGRAM_REDIRECT_TEXT}\n\n")
            self.chat_log.see(tk.END)
            self.face.set_expression("thinking")
            self.audio.speak(
                TELEGRAM_REDIRECT_TEXT,
                on_start=self.face.start_talking,
                on_end=self._tts_finish_face_state,
            )
            self.processing = False
            self.set_status("READY")
            return

        vision_keywords = ["look", "see", "show", "analyze", "watch"]
        image_path = None
        
        if any(word in text.lower() for word in vision_keywords):
            self.set_status("📸 TRANSMITTING PHOTO...")
            image_path = self.vision.save_frame(self.current_frame_img)

        self.set_status("📡 AWAITING MACBOOK...")
        self.chat_log.insert(tk.END, "Agent: ")
        
        full_reply = ""
        # Standard synchronous for-loop processing the generator
        for token in self.brain.generate_response_stream(text, image_path):
            full_reply += token
            self.chat_log.insert(tk.END, token)
            self.chat_log.see(tk.END)

        self.chat_log.insert(tk.END, "\n\n")
        emotion = self.face.get_emotion_from_text(full_reply)
        self.face.set_expression(emotion)
        self.audio.speak(
            full_reply,
            on_start=self.face.start_talking,
            on_end=self._tts_finish_face_state,
        )
        
        self.processing = False
        self.set_status("READY")

    def proactive_heartbeat_loop(self):
        """Runs in the background, checking the user's state every 60 seconds."""
        while self.running:
            time.sleep(60) 
            
            if not self.processing and self.current_frame_img:
                # 1. LOCK THE SYSTEM: Prevent voice/text inputs while checking
                self.processing = True 
                
                print("[HEARTBEAT] Analyzing user state...")
                temp_path = self.vision.save_frame(self.current_frame_img, "heartbeat_temp.jpg")
                analysis = self.brain.silent_observe(temp_path)
                
                if analysis and "SILENCE" not in analysis.upper():
                    self.set_status("💡 PROACTIVE INTERVENTION!")
                    self.chat_log.insert(tk.END, f"\nAgent (Proactive): {analysis}\n\n")
                    self.chat_log.see(tk.END)
                    emotion = self.face.get_emotion_from_text(analysis)
                    self.face.set_expression(emotion)
                    self.audio.speak(
                        analysis,
                        on_start=self.face.start_talking,
                        on_end=self._tts_finish_face_state,
                    )
                    self.brain.history.append({'role': 'assistant', 'content': analysis})
                    
                    time.sleep(5) # Brief pause before resetting status
                    
                self.set_status("READY")
                
                # 2. UNLOCK THE SYSTEM: Allow normal chat again
                self.processing = False

    def shutdown(self):
        self.running = False
        self.face.shutdown()
        self.vision.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StudyBuddyApp(root)
    root.mainloop()
