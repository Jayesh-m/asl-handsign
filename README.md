# Sign Language to Text and Speech

Real-time ASL alphabet recognition using MediaPipe hand landmarks 
and a Random Forest classifier. Achieves 98%+ accuracy on the 
ASL alphabet dataset.

## Demo
<!-- Add a GIF of your system working here — huge visual impact -->

## How it works
1. MediaPipe extracts 21 hand landmarks (63 x/y/z coordinates) per frame
2. A Random Forest classifier predicts the ASL letter
3. Gestures held for 1.5s are confirmed and spoken aloud via TTS

## Setup

### Prerequisites
- Python 3.13
- Webcam

### Installation
```bash
git clone https://github.com/yourusername/sign-language-recognition
cd sign-language-recognition
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Usage

**Step 1 — Extract landmarks from dataset**
```bash
python extract_landmarks.py
```

**Step 2 — Train the model**
```bash
python train_model.py
```

**Step 3 — Run real-time detection**
```bash
python real_time_detection.py
```

## Dataset
Uses the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) 
from Kaggle — 87,000 images across 29 classes.

## Tech stack
- MediaPipe Tasks API (hand landmark detection)
- scikit-learn (Random Forest classifier)
- OpenCV (webcam capture)
- pyttsx3 (text-to-speec)
