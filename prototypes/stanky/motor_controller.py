import cv2
import numpy as np
import threading
import time
from gpiozero import Motor, PWMOutputDevice

class MotorController:
    """Autonomous motor control using your friend's specific pin setup."""
    
    def __init__(self):
        try:
            # Match the pins from your friend's manual code
            # Right Motor (Variables named 'left' in friend's code)
            self.m_right = Motor(forward=26, backward=16)
            self.pwm_right = PWMOutputDevice(12)
            
            # Left Motor (Variables named 'right' in friend's code)
            self.m_left = Motor(forward=6, backward=5)
            self.pwm_left = PWMOutputDevice(13)
            
            print("[MOTOR] Pins 26, 16, 12 & 6, 5, 13 initialized.")
        except Exception as e:
            print(f"[MOTOR] GPIO Error: {e}")
            self.m_left = self.m_right = None

        self.state = "IDLE"  # Can be "IDLE" or "FOLLOW"
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def set_state(self, new_state: str):
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
        """Visual Servoing: Camera pixels -> PWM Speed."""
        if self.state != "FOLLOW" or self.m_left is None or pil_image is None:
            self.stop()
            return

        # 1. Image Processing
        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            # 2. Target largest face
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cx = x + (w // 2)
            face_area = w * h
            
            img_w = pil_image.width
            center_zone = img_w // 2

            # 3. Steering Logic (Using friend's PWM values)
            # If face is left, spin left. If right, spin right.
            if cx < center_zone - 60:
                self.m_left.backward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = 0.5
            elif cx > center_zone + 60:
                self.m_left.forward(); self.m_right.backward()
                self.pwm_left.value = self.pwm_right.value = 0.5
            # If centered but far, drive forward
            elif face_area < 25000:
                self.m_left.forward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = 0.5
            else:
                self.stop() # Target reached
        else:
            self.stop() # Safety: stop if user disappears

    def shutdown(self):
        self.stop()