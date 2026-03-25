LCD Face System (Software Integration Guide)

Overview
- This document describes the face display and animation system added for the embodied study buddy.
- Main files:
  - face_controller.py: Face rendering, expression state, talking animation, idle behavior, emotion mapping.
  - stanky.py: Minimal integration points in the main interaction flow.

1) What Was Implemented

FaceController structure
- Public API:
  - set_expression(emotion)
  - start_talking()
  - stop_talking()
  - set_idle()
  - get_emotion_from_text(text)
- Supported expressions:
  - neutral, happy, thinking, confused, sleepy
- Mouth states:
  - open, closed
- Rendering:
  - Uses PIL to generate placeholder faces (head, eyes, mouth) when real assets are not yet available.
  - Optional display_callback(image) for future LCD driver hookup.

Threading model
- Talk loop thread (_talk_loop):
  - Starts on start_talking().
  - Alternates mouth open/closed every ~0.22 seconds (~2 to 3 cycles/sec).
  - Stops immediately when stop_talking() is called.
- Idle loop thread (_idle_loop):
  - Starts on set_idle().
  - Uses sleepy face and periodic blink behavior.
  - Stops automatically when expression changes or talking starts.
- Synchronization:
  - threading.Lock protects expression/mouth state.
  - threading.Event signals thread stop without blocking main flow.

Emotion mapping approach
- Rule-based keyword mapping in get_emotion_from_text(text):
  - Positive language -> happy
  - Corrective language -> confused
  - Reflective/stepwise language -> thinking
  - Fallback -> neutral
- Design goal:
  - Lightweight, deterministic, no heavy dependencies.

Integration in stanky.py
- After full model response is assembled:
  - emotion = face.get_emotion_from_text(response)
  - face.set_expression(emotion)
- During TTS:
  - face.start_talking() on TTS start callback
  - face.stop_talking() on TTS end callback
- After speech:
  - face.set_idle()
- Telegram redirect:
  - Technical/screen-heavy prompts are redirected using simple keyword detection.

2) How To Test Without Hardware

Enable debug frame saving
- Already enabled in stanky.py by default:
  - FaceController(..., save_debug_frames=True, debug_output_dir="face_debug")

Where debug frames are saved
- Output directory: face_debug/
- Files:
  - latest.png (most recent frame)
  - frame_00001.png, frame_00002.png, ... (sequence over time)
- Note:
  - Folder location is relative to current working directory when stanky.py is launched.

Mock test flow
1. Run from prototypes/stanky:
   - python stanky.py
2. Trigger a normal query.
3. Verify expected transitions:
   - Expression changes based on response text.
   - During speech, mouth alternates open/closed in generated frames.
   - After speech, face enters idle (sleepy + blink cadence).
4. Trigger a technical query (for example: "show step-by-step equation derivation").
5. Verify redirect behavior:
   - Redirect message appears in chat.
   - Redirect text is spoken with mouth animation callbacks.

Expected visible behavior in debug frames
- Talking:
  - Alternating mouth shape (line/arc vs open ellipse) at a steady rate.
- Idle:
  - Mostly sleepy expression with periodic closed/open eye cycle.
- Expression updates:
  - Background and facial cues vary by emotion.

3) How To Test Once LCD Is Ready

Where to plug in LCD driver
- Use FaceController(display_callback=your_callback).
- your_callback must accept one PIL Image frame and push it to LCD.
- Recommended integration point:
  - Initialize display hardware in stanky.py during app startup.
  - Pass a callback into FaceController constructor.

Example callback shape (pseudo)
- def lcd_display_callback(image):
-     # Convert/resample as needed by your driver
-     # driver.show(image)

Expected full-system behavior with LCD
- On each assistant response:
  - Face expression updates from get_emotion_from_text(response).
- While TTS audio is playing:
  - Mouth animates open/closed continuously.
- When interaction is inactive:
  - Face enters idle behavior.
- Main loop responsiveness:
  - No blocking expected from face animation threads.

Operational notes
- Keep the display callback fast and non-blocking.
- If LCD refresh is slow, downscale frame rate in callback or throttle rendering.
- Shutdown path already calls face.shutdown() from stanky.py.
