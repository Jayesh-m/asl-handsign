import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import time

CSV_FILE     = "gesture_data.csv"
MODEL_FILE   = "gesture_model.pkl"
ENCODER_FILE = "label_encoder.pkl"

def train():
    if not os.path.exists(CSV_FILE):
        print(f"ERROR: '{CSV_FILE}' not found! Run extract_landmarks.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(CSV_FILE)
    print(f"Total samples  : {len(df)}")
    print(f"Classes found  : {sorted(df['label'].unique())}")
    print(f"\nSamples per class:")
    print(df['label'].value_counts().sort_index().to_string())

    X = df.drop('label', axis=1).values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    print(f"\nTraining samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # ── Train Model 
    print("\nTraining Random Forest...")
    start = time.time()

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"Training done in {time.time() - start:.1f} seconds")

    # ── Evaluate 
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy : {acc * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Save Model
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    with open(ENCODER_FILE, 'wb') as f:
        pickle.dump(le, f)

    print(f"Model saved     -> {MODEL_FILE}")
    print(f"Encoder saved   -> {ENCODER_FILE}")
    print("\nNow run: python real_time_detection.py")

if __name__ == "__main__":
    train()
