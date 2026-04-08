import cv2
import numpy as np
import time
from gpiozero import Motor, PWMOutputDevice

class MotorController:
    def __init__(self):
        try:
            # Pins 26, 16, 12 = Right Motor | Pins 6, 5, 13 = Left Motor
            self.m_right = Motor(forward=26, backward=16)
            self.pwm_right = PWMOutputDevice(12)
            self.m_left = Motor(forward=6, backward=5)
            self.pwm_left = PWMOutputDevice(13)
            print("[MOTOR] Pins 26,16,12 & 6,5,13 initialized.")
        except Exception as e:
            print(f"[MOTOR] GPIO Error: {e}")
            self.m_left = None

        self.state = "IDLE"
        # Load the Face Detection model
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def set_state(self, new_state: str):
        print(f"[MOTOR] Switch turned to: {new_state}")
        self.state = new_state.upper()
        if self.state == "IDLE":
            self.stop()

    def stop(self):
        if self.m_left:
            self.pwm_left.value = 0
            self.pwm_right.value = 0
            self.m_left.stop()
            self.m_right.stop()

    def process_movement(self, pil_image):
        if self.state != "FOLLOW" or self.m_left is None or pil_image is None:
            self.stop()
            return

        # 1. Image Processing
        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Detect Faces
        faces = self.cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cx = x + (w // 2)
            face_area = w * h
            
            # DIAGNOSTIC PRINT: This tells us if OpenCV actually sees you
            print(f"[MOTOR] Target Found! Center: {cx}, Area: {face_area}")

            img_w = pil_image.width
            center_zone = img_w // 2

            # 3. Movement Logic
            if cx < center_zone - 70: # User is too far left
                self.m_left.backward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = 0.6
            elif cx > center_zone + 70: # User is too far right
                self.m_left.forward(); self.m_right.backward()
                self.pwm_left.value = self.pwm_right.value = 0.6
            elif face_area < 35000: # User is far away
                self.m_left.forward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = 0.5
            else: # Stop if close enough
                print("[MOTOR] Target Reached. Braking.")
                self.stop()
        else:
            # If state is FOLLOW but no face is seen, stay still
            print("[MOTOR] Searching for face...")
            self.stop()

    def shutdown(self):
        self.stop()