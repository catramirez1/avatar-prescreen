import time
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile

import mediapipe as mp


# --- FaceMesh indices (MediaPipe) ---
# Mouth landmarks (inner-ish)
UPPER_LIP = 13
LOWER_LIP = 14

# Eye landmarks (approx eyelids)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Eye corners to estimate width
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263


def _audio_envelope(audio: np.ndarray, sr: int, fps: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    audio /= (np.max(np.abs(audio)) + 1e-9)

    frame_len = max(1, int(sr / fps))
    n_frames = max(1, int(np.ceil(len(audio) / frame_len)))
    env = np.zeros(n_frames, dtype=np.float32)

    for i in range(n_frames):
        chunk = audio[i * frame_len : (i + 1) * frame_len]
        if len(chunk) == 0:
            break
        env[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-9))

    env = env / (env.max() + 1e-9)
    env = np.convolve(env, np.ones(5) / 5, mode="same")
    return np.clip(env, 0.0, 1.0)


def _lm_to_xy(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def _draw_blink(frame, eye_top, eye_bottom, eye_left, eye_right, blink_amount: float):
    """
    blink_amount: 0=open, 1=closed
    """
    # Interpolate eyelid closing by moving top/bottom toward mid
    cx = (eye_left[0] + eye_right[0]) // 2
    cy = (eye_top[1] + eye_bottom[1]) // 2
    half_w = max(2, (eye_right[0] - eye_left[0]) // 2)
    base_h = max(2, abs(eye_bottom[1] - eye_top[1]) // 2)

    # closed height shrinks hard
    h = max(1, int(base_h * (1.0 - 0.90 * blink_amount)))

    # draw eyelid ellipse
    cv2.ellipse(frame, (cx, cy), (half_w, h), 0, 0, 360, (25, 25, 25), 2)
    if blink_amount > 0.35:
        cv2.line(frame, (cx - half_w, cy), (cx + half_w, cy), (0, 0, 0), 3)


def _animate_mouth(frame, upper, lower, openness: float, mouth_width_px: int):
    """
    openness: 0..1 controls how wide the mouth opens
    We draw a filled mouth region centered between upper/lower lip.
    """
    cx = (upper[0] + lower[0]) // 2
    cy = (upper[1] + lower[1]) // 2

    base_h = max(2, abs(lower[1] - upper[1]))
    open_h = int(base_h + openness * (base_h * 5.0))  # amplify
    open_h = max(3, min(open_h, 80))

    half_w = max(6, mouth_width_px // 2)

    # Outer mouth outline
    cv2.ellipse(frame, (cx, cy), (half_w, max(2, open_h // 2)), 0, 0, 360, (20, 20, 20), 2)
    # Inner mouth
    cv2.ellipse(frame, (cx, cy), (max(2, half_w - 6), max(2, open_h // 2 - 4)), 0, 0, 360, (0, 0, 0), -1)


def play_and_animate(wav_path, face_image_path, fps: int = 30, window_name: str = "Live Avatar"):
    wav_path = str(wav_path)
    face_image_path = str(face_image_path)

    img_bgr = cv2.imread(face_image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read face image: {face_image_path}")

    sr, audio = wavfile.read(wav_path)
    duration = len(audio) / float(sr)
    env = _audio_envelope(audio, sr, fps)

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # FaceMesh init
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    res = face_mesh.process(img_rgb)
    face_mesh.close()

    if not res.multi_face_landmarks:
        # If no landmarks, fall back to just showing the image + no creepy mouth line
        audio_proc = subprocess.Popen(["afplay", wav_path])
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        start = time.time()
        try:
            while time.time() - start < duration:
                cv2.imshow(window_name, img_bgr)
                if (cv2.waitKey(int(1000 / fps)) & 0xFF) == ord("q"):
                    break
        finally:
            try: audio_proc.terminate()
            except: pass
            cv2.destroyAllWindows()
        return

    lm = res.multi_face_landmarks[0].landmark

    # Extract key points
    upper = _lm_to_xy(lm[UPPER_LIP], w, h)
    lower = _lm_to_xy(lm[LOWER_LIP], w, h)

    lt = _lm_to_xy(lm[LEFT_EYE_TOP], w, h)
    lb = _lm_to_xy(lm[LEFT_EYE_BOTTOM], w, h)
    ll = _lm_to_xy(lm[LEFT_EYE_LEFT], w, h)
    lr = _lm_to_xy(lm[LEFT_EYE_RIGHT], w, h)

    rt = _lm_to_xy(lm[RIGHT_EYE_TOP], w, h)
    rb = _lm_to_xy(lm[RIGHT_EYE_BOTTOM], w, h)
    rl = _lm_to_xy(lm[RIGHT_EYE_LEFT], w, h)
    rr = _lm_to_xy(lm[RIGHT_EYE_RIGHT], w, h)

    mouth_width_px = max(40, int(abs(rr[0] - ll[0]) * 0.55))  # derived from eye span-ish

    # Blink scheduling
    next_blink_time = time.time() + random.uniform(2.5, 5.5)
    blink_start = None
    blink_duration = random.uniform(0.10, 0.18)

    # Play audio ONCE
    audio_proc = subprocess.Popen(["afplay", wav_path])

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    start = time.time()
    frame_i = 0

    try:
        while True:
            now = time.time()
            elapsed = now - start
            if elapsed >= duration:
                break

            a = float(env[min(frame_i, len(env) - 1)])
            frame = img_bgr.copy()

            # Blink amount
            blink_amount = 0.0
            if blink_start is None and now >= next_blink_time:
                blink_start = now
                blink_duration = random.uniform(0.10, 0.18)

            if blink_start is not None:
                t = (now - blink_start) / blink_duration
                if t >= 1.0:
                    blink_start = None
                    next_blink_time = now + random.uniform(2.5, 5.5)
                else:
                    blink_amount = 1.0 - abs(1.0 - 2.0 * t)
                    blink_amount = float(np.clip(blink_amount, 0.0, 1.0))

            # Draw blinks
            _draw_blink(frame, lt, lb, ll, lr, blink_amount)
            _draw_blink(frame, rt, rb, rl, rr, blink_amount)

            # Mouth openness from audio energy
            _animate_mouth(frame, upper, lower, a, mouth_width_px)

            cv2.imshow(window_name, frame)
            if (cv2.waitKey(int(1000 / fps)) & 0xFF) == ord("q"):
                break

            frame_i += 1

    finally:
        try:
            audio_proc.terminate()
        except Exception:
            pass
        cv2.destroyAllWindows()
