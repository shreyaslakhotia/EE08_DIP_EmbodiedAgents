#ONLY HAS PHOTO CAPTURE AND ANALYSIS FUNCTION

import ollama
import cv2
import base64

def capture_image():
    # 1. Access the MacBook Camera
    cam = cv2.VideoCapture(0)
    print("Taking photo...")
    
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        return None
    
    # 2. Convert the image to a format Ollama understands (Base64)
    _, buffer = cv2.imencode('.jpg', frame)
    image_bytes = buffer.tobytes()
    
    cam.release()
    return image_bytes

def analyze_scene():
    image_data = capture_image()
    
    if image_data:
        print("Qwen is thinking...")
        
        # 3. Send the image to Qwen 3 (Multimodal)
        response = ollama.chat(
            model='qwen3-vl:4b', # Or your specific vision-tuned model name
            messages=[{
                'role': 'user',
                'content': 'What do you see in this image? Describe the student\'s expression.',
                'images': [image_data]
            }]
        )
        
        print("\nQwen 3 Vision Analysis:")
        print(response['message']['content'])

if __name__ == "__main__":
    analyze_scene()