import whisper
import speech_recognition as sr
import ollama
import os

# 1. Load the "Ear" Model (The 'base' model is fast on MacBooks)
print("Loading Whisper model...")
stt_model = whisper.load_model("base")

def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n👂 Listening... (Speak now)")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    
    # Save audio temporarily to transcribe
    with open("temp_audio.wav", "wb") as f:
        f.write(audio.get_wav_data())
    
    # 2. Transcribe Audio to Text
    result = stt_model.transcribe("temp_audio.wav")
    text = result["text"].strip()
    os.remove("temp_audio.wav") # Cleanup
    return text

def voice_chat_loop():
    history = [{'role': 'system', 'content': 'You are a helpful voice-activated Study Buddy.'}]
    
    while True:
        # Get input via Voice instead of Typing
        user_text = listen_to_user()
        if not user_text: continue
        
        print(f"You said: {user_text}")
        if "exit" in user_text.lower(): break

        history.append({'role': 'user', 'content': user_text})

        # 3. Get response from Qwen 3
        print("Agent: ", end="", flush=True)
        response = ollama.chat(model='qwen3', messages=history, stream=True)
        
        full_reply = ""
        for chunk in response:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_reply += content
        
        history.append({'role': 'assistant', 'content': full_reply})
        print()

if __name__ == "__main__":
    voice_chat_loop()