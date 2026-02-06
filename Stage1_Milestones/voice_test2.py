import tkinter as tk
from tkinter import scrolledtext
import speech_recognition as sr
import whisper
import ollama
import threading
import os

# Load Whisper Base for MacBook balance of speed/accuracy
print("Loading Whisper...")
whisper_model = whisper.load_model("base")

class StudyBuddyAgent:
    def __init__(self, root):
        self.root = root
        self.root.title("Study Buddy v2.1")
        self.root.geometry("500x600")

        # --- UI SETUP ---
        self.label = tk.Label(root, text="Agent Status: READY", font=("Arial", 12, "bold"))
        self.label.pack(pady=10)

        self.chat_log = scrolledtext.ScrolledText(root, width=55, height=25)
        self.chat_log.pack(pady=10)

        self.stop_btn = tk.Button(root, text="EXIT PROGRAM", command=self.exit_app, bg="#FF5555")
        self.stop_btn.pack(pady=5)

        # --- LOGIC SETUP ---
        self.is_running = True
        self.is_processing = False # The "Wait Your Turn" flag
        self.history = [{'role': 'system', 'content': 'You are a concise Study Buddy assistant.'}]
        
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True 
        self.recognizer.energy_threshold = 400 # Higher = ignores background hum better

        # Start the background thread
        threading.Thread(target=self.main_agent_loop, daemon=True).start()

    def main_agent_loop(self):
        while self.is_running:
            if not self.is_processing:
                user_text = self.listen_and_transcribe()
                if user_text and len(user_text) > 3: # Ignore tiny noise/glitches
                    self.is_processing = True
                    self.update_ui_status("THINKING...")
                    self.process_with_qwen(user_text)

    def listen_and_transcribe(self):
        with sr.Microphone() as source:
            self.update_ui_status("LISTENING...")
            # Sample room noise to filter it out
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=8)
                self.update_ui_status("TRANSCRIBING...")
                
                with open("temp.wav", "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Accuracy Boost: Add 'initial_prompt' with keywords
                result = whisper_model.transcribe("temp.wav", initial_prompt="EEE, circuits, student, study, assistant")
                return result["text"].strip()
            except Exception as e:
                return None

    def process_with_qwen(self, text):
        self.update_chat(f"You: {text}")
        self.history.append({'role': 'user', 'content': text})

        try:
            # Call Qwen 3 via Ollama
            response = ollama.chat(model='qwen3-vl:4b', messages=self.history)
            reply = response['message']['content']
            
            self.update_chat(f"Agent: {reply}")
            self.history.append({'role': 'assistant', 'content': reply})
        except Exception as e:
            self.update_chat(f"SYSTEM ERROR: {e}")

        # Finish thinking and reset to listening
        self.is_processing = False

    def update_ui_status(self, status):
        self.label.config(text=f"Agent Status: {status}")

    def update_chat(self, msg):
        self.chat_log.insert(tk.END, msg + "\n\n")
        self.chat_log.see(tk.END)

    def exit_app(self):
        self.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = StudyBuddyAgent(root)
    root.mainloop()