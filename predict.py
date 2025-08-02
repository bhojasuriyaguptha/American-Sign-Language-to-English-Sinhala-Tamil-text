import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import encode_labels
import os

# Load model
model = load_model("models/asl_cnn_lstm.h5")

# Get class labels from dataset
classes = sorted(os.listdir("extracted_frames"))
_, label_encoder = encode_labels(np.array(classes))

# === 🔽 CHANGE THIS TO YOUR SAMPLE VIDEO PATH ===
sample_video_path = "data/SL/add/00964.mp4"

# Open video file and extract 20 frames
cap = cv2.VideoCapture(sample_video_path)
frames = []
while len(frames) < 20:
    ret, frame = cap.read()
    if not ret:
        break
    resized = cv2.resize(frame, (64, 64))
    normalized = resized.astype("float32") / 255.0
    frames.append(normalized)

cap.release()

# Predict if we got 20 frames
if len(frames) == 20:
    X_input = np.expand_dims(np.array(frames), axis=0)  # shape: (1, 20, 64, 64, 3)
    prediction = model.predict(X_input)
    class_index = np.argmax(prediction)
    predicted_word = label_encoder.inverse_transform([class_index])[0]
    print("🧠 Predicted word:", predicted_word)
else:
    print("❌ Could not extract 20 frames from video.")
