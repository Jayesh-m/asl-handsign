

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import csv
import os
import time
import urllib.request

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Jayesh\code\personal\sign_lang\archive\asl_alphabet_train\asl_alphabet_train"
CSV_FILE = "gesture_data.csv"
MODEL_PATH = "hand_landmarker.task"   # downloaded automatically

SAMPLES_PER_CLASS = 2000 
# ──────────────────────────────────────────────────────────────────────────────

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe hand landmarker model (~9MB)...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded!\n")
    else:
        print("Hand landmarker model already exists.\n")

def make_detector():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return HandLandmarker.create_from_options(options)

def extract_landmarks(detector, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if result.hand_landmarks:
        coords = []
        for point in result.hand_landmarks[0]:
            coords.extend([point.x, point.y, point.z])
        return coords
    return None

def extract_all():
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at:\n  {DATASET_PATH}")
        return

    download_model()
    detector = make_detector()

    classes = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])
    print(f"Found {len(classes)} gesture classes: {classes}\n")

    header = [f"x{i}" for i in range(21)] + \
             [f"y{i}" for i in range(21)] + \
             [f"z{i}" for i in range(21)] + ["label"]

    total_saved = 0
    start_time = time.time()

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cls in classes:
            cls_path = os.path.join(DATASET_PATH, cls)
            images = [
                img for img in os.listdir(cls_path)
                if img.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if SAMPLES_PER_CLASS:
                images = images[:SAMPLES_PER_CLASS]

            cls_count = 0
            for img_name in images:
                img_path = os.path.join(cls_path, img_name)
                landmarks = extract_landmarks(detector, img_path)
                if landmarks:
                    writer.writerow(landmarks + [cls])
                    cls_count += 1

            total_saved += cls_count
            elapsed = time.time() - start_time
            print(f"  [{cls:10s}] {cls_count:3d} samples  |  Total: {total_saved:5d}  |  {elapsed:.0f}s elapsed")

    print(f"\nDone! {total_saved} total samples saved to '{CSV_FILE}'")
    print("Now run: python train_model.py")

if __name__ == "__main__":
    print("=" * 55)
    print("  ASL Landmark Extraction (New MediaPipe API)")
    print(f"  Dataset : {DATASET_PATH}")
    print(f"  Limit   : {SAMPLES_PER_CLASS} samples per class")
    print(f"  Output  : {CSV_FILE}")
    print("=" * 55 + "\n")
    extract_all()