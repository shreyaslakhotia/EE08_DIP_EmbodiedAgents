import cv2
import numpy as np
import threading
import time
from gpiozero import Robot

class MotorController:
    """Handles autonomous movement using Visual Servoing (Camera as a sensor)."""
    
    def __init__(self, left_pins=(4, 14), right_pins=(17, 18)):
        # Initialize the hardware pins
        try:
            self.robot = Robot(left=left_pins, right=right_pins)
            print("[MOTOR] Pins initialized successfully.")
        except Exception as e:
            print(f"[MOTOR] Hardware Error: {e}")
            self.robot = None

        self.state = "IDLE"  # IDLE or FOLLOW
        self.running = True
        
        # Load the OpenCV face detection model
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Start the "Brainstem" reflex thread
        threading.Thread(target=self._reflex_loop, daemon=True).start()

    def set_state(self, new_state: str):
        """AI calls this to switch between IDLE and FOLLOW."""
        self.state = new_state.upper()
        if self.state == "IDLE" and self.robot:
            self.robot.stop()

    def _reflex_loop(self):
        """The autonomous loop that uses pixels to estimate distance and steering."""
        while self.running:
            # We only do the math if the AI has engaged 'FOLLOW' mode
            if self.state == "FOLLOW" and self.robot:
                # We get the frame from the main app's vision system later
                # For now, this loop waits for the robot to have a frame to look at
                pass 
            time.sleep(0.1)

    def process_movement(self, pil_image):
        """Main logic: Camera pixels -> Motor voltages."""
        if self.state != "FOLLOW" or self.robot is None or pil_image is None:
            if self.robot: self.robot.stop()
            return

        # 1. Convert PIL image to OpenCV format for detection
        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Detect Faces
        faces = self.cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            # Target the largest face (closest to robot)
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cx = x + (w // 2)
            face_area = w * h # Pixel Area (w*h) replaces a distance sensor
            
            # 3. Steering Logic (Visual Servoing)
            # Frame width is usually 480 or 640 depending on rotation
            img_w = pil_image.width
            center_zone = img_w // 2

            if cx < center_zone - 50:
                self.robot.left(speed=0.4)    # Face is on the left -> Turn Left
            elif cx > center_zone + 50:
                self.robot.right(speed=0.4)   # Face is on the right -> Turn Right
            elif face_area < 28000:           # Face is too small -> Drive Forward
                self.robot.forward(speed=0.5)
            else:
                self.robot.stop()             # Face is big enough -> Stay put
        else:
            self.robot.stop() # Lost sight of user -> Safety Stop

    def shutdown(self):
        self.running = False
        if self.robot: self.robot.stop()