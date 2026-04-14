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
from datetime import datetime
from face_controller import FaceController
# --- HOOK 1: Import the motor logic ---
from motor_controller import MotorController 

# ==========================================
# CONFIGURATION
# ==========================================
MAC_IP = "10.91.71.114"  # Replace with your MacBook's IP address
MODEL_NAME = "qwen3-vl:2b-instruct-q4_k_m"
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
        # FIX: Clean initial memory with Action Tags
        self.history = [{
            'role': 'system', 
            'content': 'You are MotivAI, an embodied Study Buddy. Provide concise answers and acknowledge movement requests naturally.'
        }]

    def generate_response_stream(self, user_text, image_path=None):
        """Yields tokens with built-in retries for robotic resilience."""
        now = datetime.now()
        # FIX: Updated system prompt to be more direct and ensure response
        system_prompt = (
            f"You are MotivAI, a helpful Study Buddy in Singapore. "
            f"It is {now.strftime('%H:%M')}. "
            f"Always answer the user's questions naturally and concisely in 1 or 2 sentences. "
            f"If asked to follow or move, acknowledge the request warmly."
        )
        
        # Ensure the system prompt is always at the start of history and up-to-date
        if self.history and self.history[0]['role'] == 'system':
            self.history[0]['content'] = system_prompt
        else:
            self.history.insert(0, {'role': 'system', 'content': system_prompt})
        
        images_b64 = []
        if image_path:
            try:
                with open(image_path, "rb") as img_file:
                    b64_string = base64.b64encode(img_file.read()).decode('utf-8')
                    images_b64.append(b64_string)
            except Exception as e:
                print(f"Image Encoding Error: {e}")

        message = {'role': 'user', 'content': user_text}
        if images_b64:
            message['images'] = images_b64
        self.history.append(message)
        
        # FIX: The Silver Bullet for Hallucinations and Repetitions
        payload = {
            "model": self.model,
            "messages": self.history,
            "stream": True,
            "keep_alive": -1,
            "options": {
                "temperature": 0.4,
                "repeat_penalty": 1.2,
                "stop": ["User:", "Agent:", "You:", "\n\nUser:"] 
            }
        }
        
        max_retries = 3
        full_reply = ""

        for attempt in range(max_retries):
            try:
                with requests.post(self.server_url, json=payload, stream=True, timeout=(10, 30)) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if "message" in chunk and "content" in chunk["message"]:
                                token = chunk["message"]["content"]
                                full_reply += token
                                yield token

                self.history.append({'role': 'assistant', 'content': full_reply})
                self._clean_history()
                return 

            except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 * (attempt + 1) 
                    yield f"\n[Signal weak... retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})]"
                    time.sleep(wait_time)
                else:
                    yield f"\n[OFFLINE: AI cannot reach the MacBook. Check the server at {MAC_IP}.]"
                    if self.history and self.history[-1]['role'] == 'user':
                        self.history.pop()

    def _clean_history(self):
        for msg in self.history:
            if 'images' in msg:
                del msg['images']

    def silent_observe(self, image_path):
        try:
            with open(image_path, "rb") as img_file:
                b64_string = base64.b64encode(img_file.read()).decode('utf-8')
            
            observation_prompt = (
                "You are an empathetic study buddy observing the user through a camera. "
                "Analyze their facial expression and body language. "
                "If they look visibly frustrated, confused, tired, or have their head in their hands, "
                "generate a very brief, friendly, supportive interruption. "
                "If they look focused, neutral, or are just reading normally, YOU MUST OUTPUT EXACTLY AND ONLY THE WORD: SILENCE."
            )

            payload = {
                "model": self.model,
                "messages": [{'role': 'user', 'content': observation_prompt, 'images': [b64_string]}],
                "stream": False 
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
            frame = self.picam2.capture_array("main")
            frame_rgb = frame[:, :, ::-1]
            img = Image.fromarray(frame_rgb)
            return img.rotate(90, expand=True)
        except Exception as e:
            print(f"Vision capture error: {e}")
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
    def __init__(self):
        print("Initializing Audio System...")
        self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5

    def listen(self, status_callback, app_instance): 
        with sr.Microphone() as source:
            status_callback("LISTENING...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)
                
                if app_instance.processing:
                    print("[AUDIO] Ignored: AI was speaking during recording.")
                    return None

                status_callback("TRANSCRIBING...")
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                segments, _ = self.whisper_model.transcribe("temp.wav", beam_size=1)
                text = " ".join([segment.text for segment in segments])
                
                if app_instance.processing:
                    return None
                    
                return text.strip()
                
            except Exception as e:
                print(f"Listen Error: {e}")
                return None

    def _clean_for_speech(self, text):
        cleaned = re.sub(r'[\*\#\[\]\(\)\`\_]', '', text)
        cleaned = re.sub(r'[^\w\s.,!?\'"\-:]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def speak(self, text, **kwargs): 
        # FIX: Removed the aggressive 'follow me' filter to allow acknowledgement
        text = re.sub(r'<[^>]+>', '', text) 
        
        cleaned_text = self._clean_for_speech(text)
        if not cleaned_text.strip():
            return 
            
        try:
            if 'on_start' in kwargs and callable(kwargs['on_start']):
                kwargs['on_start']()

            tts = gTTS(text=cleaned_text, lang='en', tld='com')
            audio_file = "speech.mp3"
            tts.save(audio_file)
            
            os.system(f"mpg123 -q {audio_file}")
            
            if 'on_end' in kwargs and callable(kwargs['on_end']):
                kwargs['on_end']()

            if os.path.exists(audio_file):
                os.remove(audio_file)
                
        except Exception as e:
            print(f"TTS Error: {e}")


# ==========================================
# 4. THE INTERFACE (Main Orchestrator)
# ==========================================
class StudyBuddyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy Client")
        self.root.geometry("800x900")

        self.brain = RemoteBrain(server_ip=MAC_IP, model=MODEL_NAME)
        self.vision = VisionSystem()
        self.audio = AudioSystem()
        
        self.face = FaceController(
            width=300,
            height=300,
            display_callback=self.update_face_ui
        )
        self.face.set_idle()

        self.motors = MotorController()

        self.running = True
        self.processing = False
        self.current_frame_img = None

        self._build_ui()
        
        self.update_video_feed()
        threading.Thread(target=self.voice_loop, daemon=True).start()
        threading.Thread(target=self.proactive_heartbeat_loop, daemon=True).start()

    def _build_ui(self):
        self.status = tk.Label(self.root, text="STATUS: READY", font=("Arial", 12, "bold"), fg="blue")
        self.status.pack(pady=5)

        self.chat_log = scrolledtext.ScrolledText(self.root, width=70, height=18)
        self.chat_log.pack(pady=10)

        self.input_frame = tk.Frame(self.root)
        self.input_frame.pack(pady=5, padx=10, fill='x')
        
        self.user_entry = tk.Entry(self.input_frame, font=("Arial", 14))
        self.user_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        # Pass the event properly to the handler
        self.user_entry.bind("<Return>", self.handle_input) 

        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(pady=5)

        self.vid_label = tk.Label(self.display_frame, bg="black")
        self.vid_label.pack(side="left", padx=10)

        self.face_label = tk.Label(self.display_frame, bg="white")
        self.face_label.pack(side="right", padx=10)

        self.send_btn = tk.Button(self.input_frame, text="SEND", command=self.handle_input)
        self.send_btn.pack(side='right')

        self.exit_btn = tk.Button(self.root, text="SHUTDOWN", command=self.shutdown, bg="red", fg="white")
        self.exit_btn.pack(pady=10)

    def update_face_ui(self, pil_img):
        img_tk = ImageTk.PhotoImage(image=pil_img)
        self.root.after(0, self._set_face_image, img_tk)

    def _set_face_image(self, img_tk):
        self.face_label.imgtk = img_tk
        self.face_label.configure(image=img_tk)

    def update_video_feed(self):
        if self.running:
            img = self.vision.get_current_frame()
            if img:
                self.current_frame_img = img.copy()
                self.motors.process_movement(img)
                if not self.processing:
                    img_gui = img.resize((300, 400))
                    imgtk = ImageTk.PhotoImage(image=img_gui)
                    self.vid_label.imgtk = imgtk
                    self.vid_label.configure(image=imgtk)
            self.root.after(100, self.update_video_feed)

    def set_status(self, msg):
        self.status.config(text=f"STATUS: {msg}")

    def voice_loop(self):
        while self.running:
            if not getattr(self, 'processing', False):
                user_text = self.audio.listen(self.set_status, self)
                if user_text and len(user_text) > 4 and not getattr(self, 'processing', False):
                    self.trigger_ai_interaction(user_text)
            time.sleep(0.3)

    def handle_input(self, event=None):
        # FIX: The Airtight Tkinter Lock
        if getattr(self, 'processing', False):
            return "break" 
            
        text = self.user_entry.get().strip()
        if text:
            self.user_entry.delete(0, tk.END)
            self.trigger_ai_interaction(text)
            
        return "break"

    def trigger_ai_interaction(self, text):
        if getattr(self, 'processing', False):
            return 
            
        self.processing = True 
        self.set_status("THINKING...")
        self.chat_log.insert(tk.END, f"You: {text}\n\n")
        self.chat_log.see(tk.END)
        threading.Thread(target=self.process_ai_stream, args=(text,), daemon=True).start()

    def _should_redirect_to_telegram(self, text):
        lower = (text or "").lower()
        redirect_keywords = ["code", "equation", "derive", "proof", "formula", "step by step", "debug"]
        return any(keyword in lower for keyword in redirect_keywords)

    def _tts_finish_face_state(self):
        self.face.stop_talking()
        self.face.set_idle()

    def process_ai_stream(self, text):
        # FIX: Robust Echo Cancellation
        if self.brain.history and self.brain.history[-1]['role'] == 'assistant':
            last_ai_words = self.brain.history[-1]['content'].strip().lower()
            incoming = text.strip().lower()
            if incoming in last_ai_words or last_ai_words in incoming:
                print(f"[DEBUG] Echo detected and dropped: {text}")
                self.processing = False
                self.set_status("READY")
                return

        if self._should_redirect_to_telegram(text):
            self.chat_log.insert(tk.END, f"Agent: {TELEGRAM_REDIRECT_TEXT}\n\n")
            self.chat_log.see(tk.END)
            self.face.set_expression("thinking")
            self.audio.speak(TELEGRAM_REDIRECT_TEXT, on_start=self.face.start_talking, on_end=self._tts_finish_face_state)
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
        for token in self.brain.generate_response_stream(text, image_path):
            full_reply += token
            self.chat_log.insert(tk.END, token)
            self.chat_log.see(tk.END)

        self.chat_log.insert(tk.END, "\n\n")

        # Motor Trigger based on user input (Hardcoded command)
        user_input_lower = text.lower()
        if "follow me" in user_input_lower:
            print("[DEBUG] 'follow me' detected. Triggering motors.")
            if self.motors:
                self.motors.set_state("FOLLOW")
        elif "stop" in user_input_lower:
            print("[DEBUG] 'stop' detected. Halting motors.")
            if self.motors:
                self.motors.set_state("IDLE")

        # FIX: Speech processing
        speech_text = full_reply.strip()

        emotion = self.face.get_emotion_from_text(full_reply)
        self.face.set_expression(emotion)
        
        self.set_status("🗣️ SPEAKING...")
        self.audio.speak(
            speech_text, 
            on_start=self.face.start_talking,
            on_end=self._tts_finish_face_state,
        )

        time.sleep(1.0) 
        self.processing = False
        self.set_status("READY")

    def proactive_heartbeat_loop(self):
        while self.running:
            time.sleep(60) 
            if not getattr(self, 'processing', False) and self.current_frame_img:
                self.processing = True 
                temp_path = self.vision.save_frame(self.current_frame_img, "heartbeat_temp.jpg")
                analysis = self.brain.silent_observe(temp_path)
                if analysis and "SILENCE" not in analysis.upper():
                    self.chat_log.insert(tk.END, f"\nAgent (Proactive): {analysis}\n\n")
                    self.audio.speak(analysis) 
                    self.brain.history.append({'role': 'assistant', 'content': analysis})
                self.processing = False 
                self.set_status("READY")

    def shutdown(self):
        self.running = False
        self.face.shutdown()
        self.motors.shutdown()
        self.vision.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StudyBuddyApp(root)
    root.mainloop()