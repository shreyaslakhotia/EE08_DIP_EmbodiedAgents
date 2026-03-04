#HAS BOTH CHATBOT AND VISION ANALYSIS FUNCTIONS

import ollama
import cv2
import os
import time

def capture_snapshot():
    """Trigger the camera, let it adjust to light, and save a bright frame."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    # --- THE FIX: RAMPING ---
    # Give the camera 1 second to physically initialize
    time.sleep(1) 
    
    # Read 30 frames and throw them away to let auto-exposure kick in
    for _ in range(30):
        cap.read() 
    
    # Now, take the final, brightened frame
    success, frame = cap.read()
    
    if not success:
        print("Error: Failed to capture bright frame.")
        cap.release()
        return None
    
    path = "current_view.jpg"
    cv2.imwrite(path, frame)
    
    cap.release()
    print("Snapshot captured with adjusted exposure.")
    return path

def run_unified_agent():
    # 1. Initialize Short-Term Memory
    history = [{'role': 'system', 'content': 'You are a Study Buddy. If an image is provided, analyze it. Otherwise, chat normally.'}]
    
    print("--- Study Buddy Agent Active ---")
    print("(Type 'exit' to quit. Use words like 'see' or 'look' to trigger the camera.)")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break

        # 2. Logic: Should I turn on the camera?
        current_images = []
        trigger_words = ['see', 'look', 'analyze', 'watch', 'view']
        
        if any(word in user_input.lower() fCor word in trigger_words):
            img_path = capture_snapshot()
            if img_path:
                current_images = [img_path]
                print("📸 [Camera Triggered] Analyzing current view...")

        # 3. Add user input to history
        history.append({'role': 'user', 'content': user_input, 'images': current_images})

        # 4. Request from Qwen 3-VL
        print("Agent: ", end="", flush=True)
        full_reply = ""
        
        # We use qwen3-vl:4b (ensure you pulled this via terminal)
        stream = ollama.chat(model='qwen3-vl:4b', messages=history, stream=True)

        for chunk in stream:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_reply += content
        print()

        # 5. Save the AI's response back to history
        history.append({'role': 'assistant', 'content': full_reply})

if __name__ == "__main__":
    run_unified_agent()
