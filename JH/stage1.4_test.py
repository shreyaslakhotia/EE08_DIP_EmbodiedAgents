import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
import whisper
import ollama
import cv2
import threading
import time
import os

# 1. SETUP BRAINS
print("Loading Whisper...")
whisper_model = whisper.load_model("base")

class MasterAgent:
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy Master v1.4")
        self.root.geometry("600x750")

        # --- UI ELEMENTS ---
        self.status = tk.Label(root, text="STATUS: READY", font=("Arial", 12, "bold"), fg="blue")
        self.status.pack(pady=10)

        self.chat_log = scrolledtext.ScrolledText(root, width=70, height=30)
        self.chat_log.pack(pady=10)

        # --- NEW: TEXT INPUT AREA ---
        self.input_frame = tk.Frame(root)
        self.input_frame.pack(pady=5, padx=10, fill='x')
        
        self.user_entry = tk.Entry(self.input_frame, font=("Arial", 14))
        self.user_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self.user_entry.bind("<Return>", self.handle_typed_input) # Bind 'Enter' key

        self.send_btn = tk.Button(self.input_frame, text="SEND", command=self.handle_typed_input)
        self.send_btn.pack(side='right')

        self.exit_btn = tk.Button(root, text="SHUTDOWN", command=self.shutdown, bg="red", fg="white")
        self.exit_btn.pack(pady=10)

        # --- STATE & LOGIC ---
        self.running = True
        self.processing = False
        self.history = [{'role': 'system', 'content': 'You are a Study Buddy with eyes and ears. Use image context if keywords like "look" or "see" are used.'}]
        self.recognizer = sr.Recognizer()
        
        # Start Voice Listening thread
        threading.Thread(target=self.voice_loop, daemon=True).start()

    def handle_typed_input(self, event=None):
        """Captures text from the entry box and sends it for processing."""
        text = self.user_entry.get().strip()
        if text and not self.processing:
            self.user_entry.delete(0, tk.END) # Clear the box
            self.processing = True
            threading.Thread(target=self.handle_interaction, args=(text,), daemon=True).start()

    def capture_vision(self):
        self.update_status("TAKING PHOTO...")
        cap = cv2.VideoCapture(0)
        time.sleep(1) 
        for _ in range(10): cap.read()
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("vision_temp.jpg", frame)
            cap.release()
            return "vision_temp.jpg"
        cap.release()
        return None

    def voice_loop(self):
        while self.running:
            if not self.processing:
                user_text = self.listen_voice()
                if user_text and len(user_text) > 4:
                    self.processing = True
                    self.handle_interaction(user_text)

    def listen_voice(self):
        with sr.Microphone() as source:
            self.update_status("LISTENING...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)
                self.update_status("TRANSCRIBING...")
                with open("temp.wav", "wb") as f: f.write(audio.get_wav_data())
                result = whisper_model.transcribe("temp.wav")
                return result["text"].strip()
            except: return None

    def handle_interaction(self, text):
        self.update_chat(f"You: {text}")
        images = []
        vision_keywords = ["look", "see", "show", "analyze", "watch", "describe my face", "what do you see"]
        
        if any(word in text.lower() for word in vision_keywords):
            img_path = self.capture_vision()
            if img_path:
                images = [img_path]

        self.update_status("THINKING...")
        self.history.append({'role': 'user', 'content': text, 'images': images})
        
        try:
            # Connect to your local Qwen3-VL
            response = ollama.chat(model='qwen3-vl:4b', messages=self.history)
            reply = response['message']['content']
            self.update_chat(f"Agent: {reply}")
            self.history.append({'role': 'assistant', 'content': reply})
        except Exception as e:
            self.update_chat(f"SYSTEM ERROR: {e}")

        self.processing = False
        self.update_status("READY")

    def update_status(self, msg):
        self.status.config(text=f"STATUS: {msg}")

    def update_chat(self, msg):
        self.chat_log.insert(tk.END, msg + "\n\n")
        self.chat_log.see(tk.END)

    def shutdown(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MasterAgent(root)
    root.mainloop()
