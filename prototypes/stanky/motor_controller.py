import cv2
import numpy as np
import time
from gpiozero import Motor, PWMOutputDevice

class MotorController:
    """Autonomous motor control for heavy chassis (Max Torque)."""
    
    def __init__(self):
        # 1. Initialize State FIRST to prevent AttributeError
        self.state = "IDLE" 
        self.is_moving = False
        self.lost_face_count = 0
        self.max_lost_frames = 10 # Buffer to prevent stuttering
        
        try:
            # RIGHT MOTOR: Pins 26, 16 | Pin 12 (PWM)
            self.m_right = Motor(forward=26, backward=16)
            self.pwm_right = PWMOutputDevice(12, frequency=1000)
            
            # LEFT MOTOR: Pins 6, 5 | Pin 13 (PWM)
            self.m_left = Motor(forward=6, backward=5)
            self.pwm_left = PWMOutputDevice(13, frequency=1000)
            
            print("[MOTOR] Pins 26,16,12 and 6,5,13 Online.")
        except Exception as e:
            print(f"[MOTOR] GPIO Setup Error: {e}")
            self.m_left = None

        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def set_state(self, new_state: str):
        self.state = new_state.upper()
        print(f"[MOTOR] State: {self.state}")
        if self.state == "IDLE":
            self.stop()

    def stop(self):
        if self.m_left:
            self.pwm_left.value = 0
            self.pwm_right.value = 0
            self.m_left.stop()
            self.m_right.stop()
        self.is_moving = False

    def process_movement(self, pil_image):
        # Safety: If state isn't FOLLOW, do nothing
        if self.state != "FOLLOW" or self.m_left is None or pil_image is None:
            self.stop()
            return

        # OpenCV Processing
        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        
        # INCREASED minNeighbors to 8 and added minSize to filter noise
        faces = self.cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=8, 
            minSize=(60, 60)
        )

        if len(faces) > 0:
            self.lost_face_count = 0
            # Pick the largest face (closest person)
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            cx = x + (w // 2)
            face_area = w * h 
            
            img_w = pil_image.width
            center_zone = img_w // 2

            # PROPORTIONAL-ISH POWER: Start at 0.8 to reduce overshooting
            power = 0.8 

            # Adjusting the deadzone to be slightly more forgiving
            if cx < center_zone - 60:
                self.m_left.backward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = power
            elif cx > center_zone + 60:
                self.m_left.forward(); self.m_right.backward()
                self.pwm_left.value = self.pwm_right.value = power
            elif face_area < 40000: 
                # Move forward if they are far away
                self.m_left.forward(); self.m_right.forward()
                self.pwm_left.value = self.pwm_right.value = 0.7 # Slower forward speed
                self.is_moving = True
            else:
                self.stop()
        else:
            # Hysteresis: Don't stop immediately if face is lost for a split second
            self.lost_face_count += 1
            if self.lost_face_count > self.max_lost_frames:
                self.stop()

    def shutdown(self):
        self.stop()