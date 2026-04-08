

import os
import pickle
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_FILE = "gesture_model.pkl"
ENCODER_FILE = "label_encoder.pkl"
MP_MODEL = "hand_landmarker.task"

# ── TTS
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)


def speak(text):
    def run():
        tts_engine.say(text)
        tts_engine.runAndWait()

    threading.Thread(target=run, daemon=True).start()


# ── Load ML Model 
def load_model():
    if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
        print("ERROR: Model files not found! Run train_model.py first.")
        return None, None
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_FILE, "rb") as f:
        le = pickle.load(f)
    print(f"Model loaded! Recognizes: {list(le.classes_)}")
    return model, le


# ── MediaPipe Detector
latest_result = {"landmarks": None}


def result_callback(result, output_image, timestamp_ms):
    if result.hand_landmarks:
        coords = []
        for point in result.hand_landmarks[0]:
            coords.extend([point.x, point.y, point.z])
        latest_result["landmarks"] = coords
        latest_result["raw"] = result.hand_landmarks[0]
    else:
        latest_result["landmarks"] = None
        latest_result["raw"] = None


def make_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MP_MODEL)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=result_callback,
    )
    return HandLandmarker.create_from_options(options)


# ── Draw hand skeleton manually ───────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def draw_landmarks(frame, raw_landmarks, w, h):
    if not raw_landmarks:
        return
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in raw_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (0, 255, 100), -1)


# ── Main
def run():
    model, le = load_model()
    if model is None:
        return

    if not os.path.exists(MP_MODEL):
        print(
            f"ERROR: {MP_MODEL} not found! Run extract_landmarks.py first (it downloads the model)."
        )
        return

    detector = make_detector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sentence = []
    last_gesture = ""
    last_spoken = 0
    hold_start = None
    HOLD_DUR = 1.5
    COOLDOWN = 2.0
    timestamp = 0

    print("\n=== Sign Language Detector Running ===")
    print("  SPACE = Speak sentence")
    print("  C     = Clear sentence")
    print("  Q     = Quit")
    print("======================================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp += 1
        detector.detect_async(mp_img, timestamp)

        landmarks = latest_result.get("landmarks")
        raw = latest_result.get("raw")

        # Draw skeleton
        draw_landmarks(frame, raw, w, h)

        predicted = None
        confidence = 0.0

        if landmarks:
            probs = model.predict_proba([landmarks])[0]
            confidence = np.max(probs)
            predicted = le.inverse_transform([np.argmax(probs)])[0]

            if confidence > 0.83:
                if predicted == last_gesture:
                    if hold_start is None:
                        hold_start = time.time()
                    held = time.time() - hold_start
                    # Progress bar
                    prog = min(int(held / HOLD_DUR * w), w)
                    cv2.rectangle(
                        frame, (0, h - 163), (prog, h - 158), (0, 255, 100), -1
                    )
                    if held >= HOLD_DUR and (time.time() - last_spoken) > COOLDOWN:
                        sentence.append(predicted)
                        speak(predicted)
                        last_spoken = time.time()
                        hold_start = None
                else:
                    last_gesture = predicted
                    hold_start = None
            else:
                last_gesture = ""
                hold_start = None
        else:
            last_gesture = ""
            hold_start = None

        # ── UI ────────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 160), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        cv2.putText(
            frame,
            "Sign Language to Text  |  Q=Quit",
            (10, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
        )

        if predicted and confidence > 0.6:
            cv2.putText(
                frame,
                f"Detected: {predicted}  ({confidence * 100:.0f}%)",
                (20, h - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 100),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Show a hand gesture...",
                (20, h - 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (150, 150, 150),
                2,
            )

        sentence_text = " ".join(sentence[-6:])
        cv2.putText(
            frame,
            f"Sentence: {sentence_text}",
            (20, h - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            "SPACE=Speak | C=Clear | Q=Quit",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (180, 180, 180),
            1,
        )

        cv2.imshow("Sign Language to Text", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        elif key == ord(" "):
            full = " ".join(sentence)
            if full:
                speak(full)
        elif key == ord("c") or key == ord("C"):
            sentence.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
