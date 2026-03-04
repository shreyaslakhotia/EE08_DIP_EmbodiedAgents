import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from picamera2 import Picamera2
from ollama import AsyncClient
from faster_whisper import WhisperModel
import speech_recognition as sr
import pyttsx3
import threading
import asyncio
import time
import os

# 1. SETUP BRAINS (Optimized for Pi 5 Architecture)
print("Loading Whisper (Tiny.en Model / INT8 Mode)...")
#int8 quantization
whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

class MasterAgent:
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy Prototype - Pi 5 Edge AI")
        self.root.geometry("800x900")

        # --- UI ELEMENTS ---
        self.status = tk.Label(root, text="STATUS: READY", font=("Arial", 12, "bold"), fg="blue")
        self.status.pack(pady=5)

        # --- NEW: LIVE VIDEO FEED ---
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
        
        # --- PICAMERA2 SETUP ---
        print("Initializing Picamera2...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.current_frame_img = None
        
        # Start the live video loop
        self.update_video_feed()

        # --- VOICE OUTPUT SETUP (Linux/eSpeak) ---
        try:
            self.engine = pyttsx3.init(driverName='espeak')
            self.engine.setProperty('rate', 160)
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if "en" in voice.id or "english" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        except Exception as e:
            print(f"Audio Driver Warning: {e}")

        # Start Voice Listening thread
        threading.Thread(target=self.voice_loop, daemon=True).start()

    def update_video_feed(self):
        """Continuously pulls frames directly from the Pi Camera sensor."""
        if self.running:
            # CRITICAL OPTIMIZATION: Stop updating the GUI video if AI is working
            # This hands 100% of the Pi 5's CPU cores back to Ollama
            if not self.processing: 
                try:
                    frame = self.picam2.capture_array("main")
                    img = Image.fromarray(frame)
                    
                    self.current_frame_img = img.copy()
                    
                    # Resize is CPU intensive. We only do it when idle.
                    img_gui = img.resize((400, 300))
                    imgtk = ImageTk.PhotoImage(image=img_gui)
                    
                    self.vid_label.imgtk = imgtk
                    self.vid_label.configure(image=imgtk)
                except Exception as e:
                    pass 
            
            # Change from 30ms (~33 FPS) to 150ms (~6 FPS)
            # You do not need a buttery smooth 60FPS feed for a Study Buddy, 
            # you need the AI to be fast.
            self.root.after(150, self.update_video_feed)

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
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)
                self.update_status("TRANSCRIBING...")
                with open("temp.wav", "wb") as f: 
                    f.write(audio.get_wav_data())
                
                # New Faster-Whisper transcription logic
                segments, info = whisper_model.transcribe("temp.wav", beam_size=1)
                
                # Rebuild the text from the chunks
                text = " ".join([segment.text for segment in segments])
                return text.strip()
            except Exception as e: 
                print(f"Voice Error: {e}")
                return None
                
    async def handle_interaction(self, text):
        self.update_chat_block(f"You: {text}")
        images = []
        vision_keywords = ["look", "see", "show", "analyze", "watch"]
        
        if any(word in text.lower() for word in vision_keywords):
            self.update_status("📸 GRABBING FRAME...")
            if self.current_frame_img is not None:
                img_path = "vision_temp.jpg"
                self.current_frame_img.save(img_path)
                images = [img_path]

        self.update_status("🧠 THINKING...")
        self.history.append({'role': 'user', 'content': text, 'images': images})
        
        try:
            self.update_chat_block("Agent: ", end_line=False)
            full_reply = ""

            # Asynchronous streaming prevents GUI freezing
            async for chunk in await AsyncClient().chat(
                model='qwen3-vl:4b', 
                messages=self.history, 
                stream=True,
                keep_alive= -1 # Keeps the model in RAM
            ):
                token = chunk['message']['content']
                full_reply += token
                self.update_chat_stream(token)

            self.update_chat_stream("\n\n") # Close the paragraph
            self.speak(full_reply)
            self.history.append({'role': 'assistant', 'content': full_reply})

            # --- THE SPEED FIX: CLEAN THE HISTORY ---
            # Remove the heavy image payload from past messages so it isn't resent
            for msg in self.history:
                if 'images' in msg:
                    del msg['images']            

        except Exception as e:
            self.update_chat_block(f"\nSYSTEM ERROR: {e}")

        self.processing = False
        self.update_status("READY")

    def update_status(self, msg):
        self.status.config(text=f"STATUS: {msg}")

    def update_chat_block(self, msg, end_line=True):
        """Standard log update for complete blocks of text."""
        suffix = "\n\n" if end_line else ""
        self.chat_log.insert(tk.END, msg + suffix)
        self.chat_log.see(tk.END)

    def update_chat_stream(self, msg):
        """Live log update for asynchronous streaming tokens."""
        self.chat_log.insert(tk.END, msg)
        self.chat_log.see(tk.END)

    def speak(self, text):
        def run_speech():
            self.engine.say(text)
            self.engine.runAndWait()
        threading.Thread(target=run_speech, daemon=True).start()

    def shutdown(self):
        self.running = False
        try:
            self.picam2.stop() # Release the hardware safely
        except: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MasterAgent(root)
    root.mainloop()