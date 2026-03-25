import os
import threading
import time
from typing import Callable, Optional

from PIL import Image, ImageDraw


class FaceController:
    """Simple face renderer/animator for Raspberry Pi LCD integration.

    Public API intended for the main app:
    - set_expression(emotion)
    - start_talking()
    - stop_talking()
    - set_idle()
    - get_emotion_from_text(text)
    """

    VALID_EXPRESSIONS = {"neutral", "happy", "thinking", "confused", "sleepy"}

    def __init__(
        self,
        width: int = 480,
        height: int = 320,
        display_callback: Optional[Callable[[Image.Image], None]] = None,
        debug_output_dir: str = "face_debug",
        save_debug_frames: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.display_callback = display_callback
        self.debug_output_dir = debug_output_dir
        self.save_debug_frames = save_debug_frames

        self._expression = "neutral"
        self._mouth_open = False
        self._is_talking = False
        self._idle_mode = False
        self._frame_index = 0

        self._lock = threading.Lock()
        self._talk_stop = threading.Event()
        self._idle_stop = threading.Event()
        self._talk_thread: Optional[threading.Thread] = None
        self._idle_thread: Optional[threading.Thread] = None

        if self.save_debug_frames:
            os.makedirs(self.debug_output_dir, exist_ok=True)

        self._render_frame()

    def set_expression(self, emotion: str) -> None:
        emotion = (emotion or "neutral").strip().lower()
        if emotion not in self.VALID_EXPRESSIONS:
            emotion = "neutral"

        with self._lock:
            self._idle_mode = False
            self._idle_stop.set()
            self._expression = emotion

        self._render_frame()

    def start_talking(self) -> None:
        with self._lock:
            self._idle_mode = False
            self._idle_stop.set()
            self._is_talking = True

        if self._talk_thread and self._talk_thread.is_alive():
            return

        self._talk_stop.clear()
        self._talk_thread = threading.Thread(target=self._talk_loop, daemon=True)
        self._talk_thread.start()

    def stop_talking(self) -> None:
        with self._lock:
            self._is_talking = False
            self._mouth_open = False

        self._talk_stop.set()
        self._render_frame()

    def set_idle(self) -> None:
        with self._lock:
            if self._is_talking:
                return
            self._idle_mode = True
            self._expression = "sleepy"
            self._mouth_open = False

        if self._idle_thread and self._idle_thread.is_alive():
            return

        self._idle_stop.clear()
        self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True)
        self._idle_thread.start()

    def get_emotion_from_text(self, text: str) -> str:
        content = (text or "").lower()

        positive_keywords = [
            "great",
            "good",
            "awesome",
            "nice",
            "well done",
            "correct",
            "perfect",
            "excellent",
            "you got it",
        ]
        reflective_keywords = [
            "let's think",
            "consider",
            "step by step",
            "first",
            "next",
            "we can",
            "reason",
            "analyze",
        ]
        corrective_keywords = [
            "not quite",
            "incorrect",
            "mistake",
            "instead",
            "try again",
            "check",
            "be careful",
            "fix",
        ]

        if any(word in content for word in positive_keywords):
            return "happy"
        if any(word in content for word in corrective_keywords):
            return "confused"
        if any(word in content for word in reflective_keywords):
            return "thinking"
        return "neutral"

    def shutdown(self) -> None:
        self._talk_stop.set()
        self._idle_stop.set()

    def _talk_loop(self) -> None:
        while not self._talk_stop.is_set():
            with self._lock:
                if not self._is_talking:
                    break
                self._mouth_open = not self._mouth_open

            self._render_frame()
            # 0.18 to 0.25 sec per mouth state gives ~2-3 open/close cycles per second.
            time.sleep(0.22)

        with self._lock:
            self._mouth_open = False
        self._render_frame()

    def _idle_loop(self) -> None:
        while not self._idle_stop.is_set():
            with self._lock:
                if not self._idle_mode or self._is_talking:
                    break
                self._expression = "sleepy"
                self._mouth_open = False

            self._render_frame(eyes_closed=True)
            if self._idle_stop.wait(0.35):
                break

            self._render_frame(eyes_closed=False)
            if self._idle_stop.wait(2.5):
                break

    def _render_frame(self, eyes_closed: bool = False) -> None:
        with self._lock:
            expression = self._expression
            mouth_open = self._mouth_open

        frame = self._draw_placeholder_face(expression, mouth_open, eyes_closed)

        if self.display_callback:
            self.display_callback(frame)

        if self.save_debug_frames:
            self._frame_index += 1
            latest_path = os.path.join(self.debug_output_dir, "latest.png")
            frame.save(latest_path)
            frame.save(os.path.join(self.debug_output_dir, f"frame_{self._frame_index:05d}.png"))

    def _draw_placeholder_face(
        self,
        expression: str,
        mouth_open: bool,
        eyes_closed: bool,
    ) -> Image.Image:
        bg_map = {
            "neutral": "#ECEFF1",
            "happy": "#FFF8E1",
            "thinking": "#E3F2FD",
            "confused": "#FFEBEE",
            "sleepy": "#E8EAF6",
        }

        image = Image.new("RGB", (self.width, self.height), bg_map.get(expression, "#ECEFF1"))
        draw = ImageDraw.Draw(image)

        cx = self.width // 2
        cy = self.height // 2
        face_r = min(self.width, self.height) // 3

        # Head outline
        draw.ellipse(
            (cx - face_r, cy - face_r, cx + face_r, cy + face_r),
            fill="#FFFDE7",
            outline="#37474F",
            width=5,
        )

        # Eyes
        ex = face_r // 2
        ey = face_r // 3
        eye_w = max(18, face_r // 6)
        eye_h = max(10, face_r // 8)

        left_eye = (cx - ex - eye_w, cy - ey - eye_h, cx - ex + eye_w, cy - ey + eye_h)
        right_eye = (cx + ex - eye_w, cy - ey - eye_h, cx + ex + eye_w, cy - ey + eye_h)

        if eyes_closed or expression == "sleepy":
            draw.line((left_eye[0], cy - ey, left_eye[2], cy - ey), fill="#263238", width=4)
            draw.line((right_eye[0], cy - ey, right_eye[2], cy - ey), fill="#263238", width=4)
        elif expression == "thinking":
            draw.ellipse(left_eye, fill="#263238")
            draw.ellipse(right_eye, fill="#263238")
            draw.arc((cx + ex - 12, cy - ey - 24, cx + ex + 26, cy - ey + 10), 200, 350, fill="#263238", width=3)
        elif expression == "confused":
            draw.ellipse(left_eye, fill="#263238")
            draw.ellipse(right_eye, fill="#263238")
            draw.line((left_eye[0], left_eye[1] - 12, left_eye[2], left_eye[1] - 18), fill="#263238", width=3)
            draw.line((right_eye[0], right_eye[1] - 18, right_eye[2], right_eye[1] - 12), fill="#263238", width=3)
        else:
            draw.ellipse(left_eye, fill="#263238")
            draw.ellipse(right_eye, fill="#263238")

        # Mouth
        mouth_w = face_r
        mouth_top = cy + face_r // 3
        if mouth_open:
            draw.ellipse(
                (
                    cx - mouth_w // 4,
                    mouth_top,
                    cx + mouth_w // 4,
                    mouth_top + max(18, face_r // 5),
                ),
                fill="#B71C1C",
                outline="#4E342E",
                width=3,
            )
        else:
            if expression == "happy":
                draw.arc((cx - mouth_w // 2, mouth_top - 10, cx + mouth_w // 2, mouth_top + 35), 10, 170, fill="#4E342E", width=4)
            elif expression == "confused":
                draw.arc((cx - mouth_w // 2, mouth_top + 5, cx + mouth_w // 2, mouth_top + 40), 190, 350, fill="#4E342E", width=4)
            else:
                draw.line((cx - mouth_w // 3, mouth_top + 12, cx + mouth_w // 3, mouth_top + 12), fill="#4E342E", width=4)

        return image
