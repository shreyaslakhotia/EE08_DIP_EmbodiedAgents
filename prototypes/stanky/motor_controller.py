import cv2
import numpy as np
import time
from gpiozero import Motor, PWMOutputDevice

class MotorController:
    def __init__(self):
        try:
            # Match your friend's exact hardware setup
            self.m_right = Motor(forward=26, backward=16)
            self.pwm_right = PWMOutputDevice(12)
            self.m_left = Motor(forward=6, backward=5)
            self.pwm_left = PWMOutputDevice(13)
            print("[MOTOR] Hardware initialized on Pins 26,16,12 and 6,5,13")
        except Exception as e:
            print(f"[MOTOR] GPIO Critical Error: {e}")
            self.m_left = None

        self.state = "IDLE"
        # Load the face model - using the absolute path to avoid Pi errors
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def set_state(self, new_state: str):
        self.state = new_state.upper()
        print(f"\n[MOTOR] STATE CHANGED TO: {self.state}")
        
        # --- HARDWARE OVERRIDE TEST ---
        # If the AI says follow, jumpstart the motors for 0.5s 
        # to prove the code can talk to the wheels.
        if self.state == "FOLLOW" and self.m_left:
            print("[MOTOR] JOLT TEST: Spinning wheels for 0.5s...")
            self.m_left.forward()
            self.m_right.forward()
            self.pwm_left.value = 0.6
            self.pwm_right.value = 0.6
            time.sleep(0.5)
            self.stop()
        else:
            self.stop()

    def stop(self):
        if self.m_left:
            self.pwm_left.value = 0
            self.pwm_right.value = 0
            self.m_left.stop()
            self.m_right.stop()

    def process_movement(self, pil_image):
        if self.state != "FOLLOW" or self.m_left is None or pil_image is None:
            return

        # 1. Image Conversion
        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        # 2. Face Detection
        faces = self.cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # Pick the largest face
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cx = x + (w // 2)
            face_area = w * h
            
            print(f"[MOTOR] I see you! Center X: {cx} | Area: {face_area}")

            img_w = pil_image.width
            center_zone = img_w // 2

            # 3. Steering Logic (Matching friend's speed levels)
            if cx < center_zone - 80:
                print("[MOTOR] Steering Left")
                self.m_left.backward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = 0.6
            elif cx > center_zone + 80:
                print("[MOTOR] Steering Right")
                self.m_left.forward(); self.m_right.backward()
                self.pwm_left.value = self.pwm_right.value = 0.6
            elif face_area < 40000: # If face is small, you are far away
                print("[MOTOR] Driving Forward")
                self.m_left.forward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = 0.5
            else:
                print("[MOTOR] Within range. Stopping.")
                self.stop()
        else:
            # This is likely where it's getting stuck!
            # If the camera rotation is wrong, it never sees a face.
            self.stop()

    def shutdown(self):
        self.stop()