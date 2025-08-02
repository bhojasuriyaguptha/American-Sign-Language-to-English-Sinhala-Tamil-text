import numpy as np
from models.cnn_lstm_model import build_cnn_lstm_model
from utils.preprocessing import load_data_from_frames, encode_labels, one_hot_encode
from sklearn.model_selection import train_test_split
import os

# Load data
X, y = load_data_from_frames("extracted_frames", frame_size=(64, 64), frame_limit=20)
y_encoded, label_encoder = encode_labels(y)
y_onehot = one_hot_encode(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Build and train model
model = build_cnn_lstm_model(input_shape=X_train.shape[1:], num_classes=y_onehot.shape[1])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=20, batch_size=4)

# Save model
model.save("models/asl_cnn_lstm.h5")
