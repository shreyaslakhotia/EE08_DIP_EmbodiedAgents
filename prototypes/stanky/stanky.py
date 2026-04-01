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
MAC_IP = "10.91.143.5"  # Replace with your MacBook's IP address
MODEL_NAME = "stanky3"
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
                    yield f"\n[OFFLINE: AI cannot reach the MacBook. Check the server at {MAC_IP}.]"
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

            # If the camera already gives RGB, skip swap; if BGR, convert.
            # Many PiCamera configs produce RGB, so this is safe either way.
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
    """Manages text-to-speech and speech-to-text."""
    def __init__(self):
        print("Initializing Audio System...")
        self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5

    def listen(self, status_callback, app_instance): 
        """Listens for audio, but ignores it if the AI is currently speaking."""
        with sr.Microphone() as source:
            status_callback("LISTENING...")
            # Help the mic ignore background hum
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            try:
                # Capture the audio from the mic
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)
                
                # --- THE LOCK CHECK ---
                # If the AI started talking (heartbeat or reply) while we were 
                # recording, discard this audio immediately.
                if app_instance.processing:
                    print("[AUDIO] Ignored: AI was speaking during recording.")
                    return None

                status_callback("TRANSCRIBING...")
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                segments, _ = self.whisper_model.transcribe("temp.wav", beam_size=1)
                text = " ".join([segment.text for segment in segments])
                
                # --- THE FINAL LOCK CHECK ---
                # Double-check one last time before returning the text.
                if app_instance.processing:
                    return None
                    
                return text.strip()
                
            except Exception as e:
                print(f"Listen Error: {e}")
                return None

    def _clean_for_speech(self, text):
        """Sanitizes LLM markdown and emojis so the TTS doesn't read them."""
        cleaned = re.sub(r'[\*\#\[\]\(\)\`\_]', '', text)
        cleaned = re.sub(r'[^\w\s.,!?\'"\-:]', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def speak(self, text, **kwargs): 
        """Blocks the thread until speech finishes, handling start/end callbacks."""
        cleaned_text = self._clean_for_speech(text)
        if not cleaned_text:
            return 
            
        try:
            # 1. Run 'on_start' (e.g., start moving the robot's mouth)
            if 'on_start' in kwargs and callable(kwargs['on_start']):
                kwargs['on_start']()

            tts = gTTS(text=cleaned_text, lang='en', tld='com')
            audio_file = "speech.mp3"
            tts.save(audio_file)
            
            # 2. Play audio (This blocks the thread until finished)
            os.system(f"mpg123 -q {audio_file}")
            
            # 3. Run 'on_end' (e.g., stop moving the mouth)
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
    """The main Tkinter GUI that glues all systems together synchronously."""
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy Edge Client")
        self.root.geometry("800x900")

        self.brain = RemoteBrain(server_ip=MAC_IP, model=MODEL_NAME)
        self.vision = VisionSystem()
        self.audio = AudioSystem()
        # Initialize the FaceController and LINK it to the UI bridge
        self.face = FaceController(
            width=300,  # Smaller width to fit side-by-side with camera
            height=300,
            display_callback=self.update_face_ui, # THIS IS THE KEY LINK
            debug_output_dir="face_debug",
            save_debug_frames=False, # Set to False once working to save SD card life
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

        # Create a frame to hold both "Eyes" (Camera) and "Face" (Expression)
        self.display_frame = tk.Frame(self.root)
        self.display_frame.pack(pady=5)

        # Camera Label (Eyes)
        self.vid_label = tk.Label(self.display_frame, bg="black")
        self.vid_label.pack(side="left", padx=10)

        # NEW: Face Label (Expression)
        self.face_label = tk.Label(self.display_frame, bg="white")
        self.face_label.pack(side="right", padx=10)

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
                    img_gui = img.resize((300, 400))
                    imgtk = ImageTk.PhotoImage(image=img_gui)
                    self.vid_label.imgtk = imgtk
                    self.vid_label.configure(image=imgtk)
            
            self.root.after(150, self.update_video_feed)

    def set_status(self, msg):
        self.status.config(text=f"STATUS: {msg}")

    def voice_loop(self):
        while self.running:
            if not self.processing:
                # We pass 'self' (the StudyBuddyApp) as the 3rd argument
                user_text = self.audio.listen(self.set_status, self)
                
                if user_text and len(user_text) > 4 and not self.processing:
                    self.trigger_ai_interaction(user_text)
            
            time.sleep(0.3) # Give the CPU a tiny breather

    def handle_input(self):
            if self.processing:
                return # Do nothing if already thinking
                
            text = self.user_entry.get().strip()
            if text:
                self.user_entry.delete(0, tk.END) # Clear the box IMMEDIATELY
                self.trigger_ai_interaction(text)

    def trigger_ai_interaction(self, text):
            """Bridges the Tkinter UI to the background network thread."""
            # --- FIX 1: LOCK IMMEDIATELY ---
            if self.processing:
                return # Block any second trigger attempts
                
            self.processing = True 
            self.set_status("THINKING...")
            
            self.chat_log.insert(tk.END, f"You: {text}\n\n")
            self.chat_log.see(tk.END)

            # Now start the thread, but the flag is already True
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
        
        # --- FIX 1: EARLY ECHO CHECK ---
        # If the user text is exactly what the AI just said, stop immediately.
        if self.brain.history and self.brain.history[-1]['role'] == 'assistant':
            last_ai_words = self.brain.history[-1]['content'].strip().lower()
            if text.strip().lower() == last_ai_words:
                print("[DEBUG] Blocked an echo repeat at the start.")
                self.processing = False
                self.set_status("READY")
                return

        # --- FIX 2: TELEGRAM REDIRECT ---
        if self._should_redirect_to_telegram(text):
            self.chat_log.insert(tk.END, f"Agent: {TELEGRAM_REDIRECT_TEXT}\n\n")
            self.chat_log.see(tk.END)
            self.face.set_expression("thinking")
            # Speak the redirect messageimport os

    def update_face_ui(self, pil_img):
        """Thread-safe callback that the FaceController calls to push new frames."""
        # Convert the PIL image from FaceController into a Tkinter-compatible format
        img_tk = ImageTk.PhotoImage(image=pil_img)
        
        # We use .after(0, ...) to force the update to happen on the main GUI thread
        # This prevents the "RuntimeError: main thread is not in main loop" crash
        self.root.after(0, self._set_face_image, img_tk)

    def _set_face_image(self, img_tk):
        """Internal helper to actually update the label."""
        self.face_label.imgtk = img_tk
        self.face_label.configure(image=img_tk)

import threading
import time
from typing import Callable, Optional

from PIL import Image, ImageDraw


class FaceController:
    """Simple face renderer/animator for Raspberry Pi LCD integration.

    Public API intended for the main app:
    - set_expression(emotion)
    - start_talking()
    - stop_talking()
    - set_idle()
    - get_emotion_from_text(text)
    """

    VALID_EXPRESSIONS = {"neutral", "happy", "thinking", "confused", "sleepy"}

    def __init__(
        self,
        width: int = 480,
        height: int = 320,
        display_callback: Optional[Callable[[Image.Image], None]] = None,
        debug_output_dir: str = "face_debug",
        save_debug_frames: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.display_callback = display_callback
        self.debug_output_dir = debug_output_dir
        self.save_debug_frames = save_debug_frames

        self._expression = "neutral"
        self._mouth_open = False
        self._is_talking = False
        self._idle_mode = False
        self._frame_index = 0

        self._lock = threading.Lock()
        self._talk_stop = threading.Event()
        self._idle_stop = threading.Event()
        self._talk_thread: Optional[threading.Thread] = None
        self._idle_thread: Optional[threading.Thread] = None

        if self.save_debug_frames:
            os.makedirs(self.debug_output_dir, exist_ok=True)

        self._render_frame()

    def set_expression(self, emotion: str) -> None:
        emotion = (emotion or "neutral").strip().lower()
        if emotion not in self.VALID_EXPRESSIONS:
            emotion = "neutral"

        with self._lock:
            self._idle_mode = False
            self._idle_stop.set()
            self._expression = emotion

        self._render_frame()

    def start_talking(self) -> None:
        with self._lock:
            self._idle_mode = False
            self._idle_stop.set()
            self._is_talking = True

        if self._talk_thread and self._talk_thread.is_alive():
            return

        self._talk_stop.clear()
        self._talk_thread = threading.Thread(target=self._talk_loop, daemon=True)
        self._talk_thread.start()

    def stop_talking(self) -> None:
        with self._lock:
            self._is_talking = False
            self._mouth_open = False

        self._talk_stop.set()
        self._render_frame()

    def set_idle(self) -> None:
        with self._lock:
            if self._is_talking:
                return
            self._idle_mode = True
            self._expression = "sleepy"
            self._mouth_open = False

        if self._idle_thread and self._idle_thread.is_alive():
            return

        self._idle_stop.clear()
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()

    def get_emotion_from_text(self, text: str) -> str:
        content = (text or "").lower()

        positive_keywords = [
            "great",
            "good",
            "awesome",
            "nice",
            "well done",
            "correct",
            "perfect",
            "excellent",
            "you got it",
        ]
        reflective_keywords = [
            "let's think",
            "consider",
            "step by step",
            "first",
            "next",
            "we can",
            "reason",
            "analyze",
        ]
        corrective_keywords = [
            "not quite",
            "incorrect",
            "mistake",
            "instead",
            "try again",
            "check",
            "be careful",
            "fix",
        ]

        if any(word in content for word in positive_keywords):
            return "happy"
        if any(word in content for word in corrective_keywords):
            return "confused"
        if any(word in content for word in reflective_keywords):
            return "thinking"
        return "neutral"

    def shutdown(self) -> None:
        self._talk_stop.set()
        self._idle_stop.set()

    def _talk_loop(self) -> None:
        while not self._talk_stop.is_set():
            with self._lock:
                if not self._is_talking:
                    break
                self._mouth_open = not self._mouth_open

            self._render_frame()
            # 0.18 to 0.25 sec per mouth state gives ~2-3 open/close cycles per second.
            time.sleep(0.22)

        with self._lock:
            self._mouth_open = False
        self._render_frame()

    def _idle_loop(self) -> None:
        while not self._idle_stop.is_set():
            with self._lock:
                if not self._idle_mode or self._is_talking:
                    break
                self._expression = "sleepy"
                self._mouth_open = False

            self._render_frame(eyes_closed=True)
            if self._idle_stop.wait(0.35):
                break

            self._render_frame(eyes_closed=False)
            if self._idle_stop.wait(2.5):
                break

    def _render_frame(self, eyes_closed: bool = False) -> None:
        with self._lock:
            expression = self._expression
            mouth_open = self._mouth_open

        frame = self._draw_placeholder_face(expression, mouth_open, eyes_closed)

        if self.display_callback:
            self.display_callback(frame)

        if self.save_debug_frames:
            self._frame_index += 1
            latest_path = os.path.join(self.debug_output_dir, "latest.png")
            frame.save(latest_path)
            frame.save(os.path.join(self.debug_output_dir, f"frame_{self._frame_index:05d}.png"))

    def _draw_placeholder_face(
        self,
        expression: str,
        mouth_open: bool,
        eyes_closed: bool,
    ) -> Image.Image:
        bg_map = {
            "neutral": "#ECEFF1",
            "happy": "#FFF8E1",
            "thinking": "#E3F2FD",
            "confused": "#FFEBEE",
            "sleepy": "#E8EAF6",
        }

        image = Image.new("RGB", (self.width, self.height), bg_map.get(expression, "#ECEFF1"))
        draw = ImageDraw.Draw(image)

        cx = self.width // 2
        cy = self.height // 2
        face_r = min(self.width, self.height) // 3

        # Head outline
        draw.ellipse(
            (cx - face_r, cy - face_r, cx + face_r, cy + face_r),
            fill="#FFFDE7",
            outline="#37474F",
            width=5,
        )

        # Eyes
        ex = face_r // 2
        ey = face_r // 3
        eye_w = max(18, face_r // 6)
        eye_h = max(10, face_r // 8)

        left_eye = (cx - ex - eye_w, cy - ey - eye_h, cx - ex + eye_w, cy - ey + eye_h)
        right_eye = (cx + ex - eye_w, cy - ey - eye_h, cx + ex + eye_w, cy - ey + eye_h)

        if eyes_closed or expression == "sleepy":
            draw.line((left_eye[0], cy - ey, left_eye[2], cy - ey), fill="#263238", width=4)
            draw.line((right_eye[0], cy - ey, right_eye[2], cy - ey), fill="#263238", width=4)
        elif expression == "thinking":
            draw.ellipse(left_eye, fill="#263238")
            draw.ellipse(right_eye, fill="#263238")
            draw.arc((cx + ex - 12, cy - ey - 24, cx + ex + 26, cy - ey + 10), 200, 350, fill="#263238", width=3)
        elif expression == "confused":
            draw.ellipse(left_eye, fill="#263238")
            draw.ellipse(right_eye, fill="#263238")
            draw.line((left_eye[0], left_eye[1] - 12, left_eye[2], left_eye[1] - 18), fill="#263238", width=3)
            draw.line((right_eye[0], right_eye[1] - 18, right_eye[2], right_eye[1] - 12), fill="#263238", width=3)
        else:
            draw.ellipse(left_eye, fill="#263238")
            draw.ellipse(right_eye, fill="#263238")

        # Mouth
        mouth_w = face_r
        mouth_top = cy + face_r // 3
        if mouth_open:
            draw.ellipse(
                (
                    cx - mouth_w // 4,
                    mouth_top,
                    cx + mouth_w // 4,
                    mouth_top + max(18, face_r // 5),
                ),
                fill="#B71C1C",
                outline="#4E342E",
                width=3,
            )
        else:
            if expression == "happy":
                draw.arc((cx - mouth_w // 2, mouth_top - 10, cx + mouth_w // 2, mouth_top + 35), 10, 170, fill="#4E342E", width=4)
            elif expression == "confused":
                draw.arc((cx - mouth_w // 2, mouth_top + 5, cx + mouth_w // 2, mouth_top + 40), 190, 350, fill="#4E342E", width=4)
            else:
                draw.line((cx - mouth_w // 3, mouth_top + 12, cx + mouth_w // 3, mouth_top + 12), fill="#4E342E", width=4)

        return image

            self.audio.speak(
                TELEGRAM_REDIRECT_TEXT,
                on_start=self.face.start_talking,
                on_end=self._tts_finish_face_state,
            )
            time.sleep(1.0) # Echo cool-down
            self.processing = False
            self.set_status("READY")
            return

        # --- FIX 3: VISION CHECK ---
        vision_keywords = ["look", "see", "show", "analyze", "watch"]
        image_path = None
        if any(word in text.lower() for word in vision_keywords):
            self.set_status("📸 TRANSMITTING PHOTO...")
            image_path = self.vision.save_frame(self.current_frame_img)

        # --- FIX 4: GENERATE RESPONSE ---
        self.set_status("📡 AWAITING MACBOOK...")
        self.chat_log.insert(tk.END, "Agent: ")
        
        full_reply = ""
        for token in self.brain.generate_response_stream(text, image_path):
            full_reply += token
            self.chat_log.insert(tk.END, token)
            self.chat_log.see(tk.END)

        self.chat_log.insert(tk.END, "\n\n")

        # --- FIX 5: SINGLE SPEAK CALL ---
        # Set the face expression based on the reply
        emotion = self.face.get_emotion_from_text(full_reply)
        self.face.set_expression(emotion)
        self.set_status("🗣️ SPEAKING...")

        # We call speak ONCE. It handles the mouth start/end and the audio.
        self.audio.speak(
            full_reply,
            on_start=self.face.start_talking,
            on_end=self._tts_finish_face_state,
        )

        # --- FIX 6: COOL-DOWN ---
        # Wait 1 second for the room to go quiet before allowing the mic to wake up
        time.sleep(1.0)
        self.processing = False
        self.set_status("READY")

    def proactive_heartbeat_loop(self):
            while self.running:
                time.sleep(60) 
                if not self.processing and self.current_frame_img:
                    self.processing = True # LOCK THE MIC
                    
                    temp_path = self.vision.save_frame(self.current_frame_img, "heartbeat_temp.jpg")
                    analysis = self.brain.silent_observe(temp_path)
                    
                    if analysis and "SILENCE" not in analysis.upper():
                        self.set_status("💡 INTERVENING...")
                        self.chat_log.insert(tk.END, f"\nAgent (Proactive): {analysis}\n\n")
                        
                        # Blocks here until speech is done
                        self.audio.speak(analysis) 
                        
                        self.brain.history.append({'role': 'assistant', 'content': analysis})

                    self.processing = False # UNLOCK THE MIC
                    self.set_status("READY")

    def shutdown(self):
        self.running = False
        self.face.shutdown()
        self.vision.shutdown()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = StudyBuddyApp(root)
    root.mainloop()
