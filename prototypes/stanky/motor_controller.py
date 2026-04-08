import cv2
import numpy as np
import time
from gpiozero import Motor, PWMOutputDevice

class MotorController:
    def __init__(self):
        try:
            self.m_right = Motor(forward=26, backward=16)
            # Set frequency to 1000Hz for better torque response
            self.pwm_right = PWMOutputDevice(12, frequency=1000) 
            
            self.m_left = Motor(forward=6, backward=5)
            self.pwm_left = PWMOutputDevice(13, frequency=1000)
            print("[MOTOR] Hardware Initialized at 1000Hz")
        except:
            self.m_left = None

    def set_state(self, new_state: str):
        self.state = new_state.upper()
        print(f"[MOTOR] STATE: {self.state}")
        
        if self.state == "FOLLOW":
            # BRUTE FORCE TEST: Ignore camera, just drive forward at 100% power
            print("[MOTOR] CRITICAL TEST: Driving 100% power for 2 seconds...")
            self.m_left.forward()
            self.m_right.forward()
            self.pwm_left.value = 1.0
            self.pwm_right.value = 1.0
            time.sleep(2.0)
            self.stop()

    def stop(self):
        if self.m_left:
            self.pwm_left.value = 0
            self.pwm_right.value = 0
            self.m_left.stop()
            self.m_right.stop()
        self.is_moving = False

    def process_movement(self, pil_image):
        if self.state != "FOLLOW" or self.m_left is None or pil_image is None:
            self.stop()
            return

        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) > 0:
            self.lost_face_count = 0 # Reset the "lost" counter
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cx = x + (w // 2)
            face_area = w * h
            
            img_w = pil_image.width
            center_zone = img_w // 2

            # --- EEE FIX: KICKSTART & POWER INCREASE ---
            # If we were stopped, give a 1.0 "Kick" to break friction
            current_power = 0.8 if self.is_moving else 1.0 
            self.is_moving = True

            if cx < center_zone - 65:
                self.m_left.backward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = current_power
            elif cx > center_zone + 65:
                self.m_left.forward(); self.m_right.backward()
                self.pwm_left.value = self.pwm_right.value = current_power
            elif face_area < 30000: 
                self.m_left.forward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = current_power
            else:
                self.stop()
        else:
            # --- NEW: BUFFER LOGIC ---
            # Don't stop immediately. Increment counter.
            self.lost_face_count += 1
            if self.lost_face_count >= self.max_lost_frames:
                self.stop()

    def shutdown(self):
        self.stop()