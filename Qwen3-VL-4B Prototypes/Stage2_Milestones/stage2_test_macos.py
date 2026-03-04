import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from ollama import AsyncClient
import speech_recognition as sr
import whisper
import pyttsx3
import threading
import asyncio
import cv2
import os

# 1. SETUP BRAINS (MacOS Optimized)
print("Loading Whisper (Base Model)...")
whisper_model = whisper.load_model("base")

class MasterAgent:
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy Prototype - macOS Environment")
        self.root.geometry("800x900")

        # --- UI ELEMENTS ---
        self.status = tk.Label(root, text="STATUS: READY", font=("Arial", 12, "bold"), fg="blue")
        self.status.pack(pady=5)

        # --- LIVE VIDEO FEED ---
        self.vid_label = tk.Label(root, bg="black")
        self.vid_label.pack(pady=5)

        self.chat_log = scrolledtext.ScrolledText(root, width=70, height=18)
        self.chat_log.pack(pady=10)

        # --- TEXT INPUT AREA ---
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(pady=5, padx=10, fill='x')
        
        self.user_entry = tk.Entry(self.input_frame, font=("Arial", 14))
        self.user_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.user_entry.bind("<Return>", self.handle_typed_input)

        self.send_btn = tk.Button(self.input_frame, text="SEND", command=self.handle_typed_input)
        self.send_btn.pack(side='right')

        self.exit_btn = tk.Button(root, text="SHUTDOWN", command=self.shutdown, bg="red", fg="white")
        self.exit_btn.pack(pady=10)

        # --- STATE & LOGIC ---
        self.running = True
        self.processing = False
        self.history = [{'role': 'system', 'content': 'You are an embodied Study Buddy. Provide concise, helpful answers.'}]
        self.recognizer = sr.Recognizer()
        
        # --- OPENCV MAC SETUP ---
        print("Initializing FaceTime Camera...")
        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        
        # Start the live video loop
        self.update_video_feed()

        # --- VOICE OUTPUT SETUP (macOS Native) ---
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 170)
        # Target Apple's premium voice
        self.engine.setProperty('voice', 'com.apple.voice.premium.en-US.Ava')

        # Start Voice Listening thread
        threading.Thread(target=self.voice_loop, daemon=True).start()

    def update_video_feed(self):
        """Continuously pulls frames from the MacBook camera."""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # Store the high-res BGR frame in memory for the AI
                self.current_frame = frame.copy()
                
                # Convert BGR (OpenCV) to RGB (Tkinter) specifically for the GUI
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize for the Tkinter display
                img_gui = img.resize((400, 300))
                imgtk = ImageTk.PhotoImage(image=img_gui)
                
                self.vid_label.imgtk = imgtk
                self.vid_label.configure(image=imgtk)
            
            # Loop at ~30 FPS
            self.root.after(30, self.update_video_feed)

    def trigger_interaction(self, text):
        """Safely bridges the synchronous GUI with the asynchronous AI."""
        self.processing = True
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.handle_interaction(text))
            loop.close()
        threading.Thread(target=run_async, daemon=True).start()

    def handle_typed_input(self, event=None):
        text = self.user_entry.get().strip()
        if text and not self.processing:
            self.user_entry.delete(0, tk.END)
            self.trigger_interaction(text)

    def voice_loop(self):
        while self.running:
            if not self.processing:
                user_text = self.listen_voice()
                if user_text and len(user_text) > 4:
                    self.trigger_interaction(user_text)

    def listen_voice(self):
        with sr.Microphone() as source:
            self.update_status("LISTENING...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)
                self.update_status("TRANSCRIBING...")
                with open("temp.wav", "wb") as f: 
                    f.write(audio.get_wav_data())
                
                # Standard transcription for Mac
                result = whisper_model.transcribe("temp.wav")
                return result["text"].strip()
            except: 
                return None

    async def handle_interaction(self, text):
        self.update_chat_block(f"You: {text}")
        images = []
        vision_keywords = ["look", "see", "show", "analyze", "watch"]
        
        if any(word in text.lower() for word in vision_keywords):
            self.update_status("📸 GRABBING FRAME...")
            if self.current_frame is not None:
                img_path = "vision_temp.jpg"
                # OpenCV uses imwrite to save the BGR frame directly
                cv2.imwrite(img_path, self.current_frame)
                images = [img_path]

        self.update_status("🧠 THINKING...")
        self.history.append({'role': 'user', 'content': text, 'images': images})
        
        try:
            self.update_chat_block("Agent: ", end_line=False)
            full_reply = ""

            # Asynchronous streaming
            async for chunk in await AsyncClient().chat(
                model='qwen3-vl:4b', 
                messages=self.history, 
                stream=True
            ):
                token = chunk['message']['content']
                full_reply += token
                self.update_chat_stream(token)

            self.update_chat_stream("\n\n") 
            self.speak(full_reply)
            self.history.append({'role': 'assistant', 'content': full_reply})

        except Exception as e:
            self.update_chat_block(f"\nSYSTEM ERROR: {e}")

        self.processing = False
        self.update_status("READY")

    def update_status(self, msg):
        self.status.config(text=f"STATUS: {msg}")

    def update_chat_block(self, msg, end_line=True):
        suffix = "\n\n" if end_line else ""
        self.chat_log.insert(tk.END, msg + suffix)
        self.chat_log.see(tk.END)

    def update_chat_stream(self, msg):
        self.chat_log.insert(tk.END, msg)
        self.chat_log.see(tk.END)
        self.root.update_idletasks() 

    def speak(self, text):
        def run_speech():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=run_speech, daemon=True).start()

    def shutdown(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release() # Release MacBook camera
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MasterAgent(root)
    root.mainloop()