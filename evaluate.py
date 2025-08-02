import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import load_data_from_frames, encode_labels
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os

# Load test data
X, y = load_data_from_frames("extracted_frames", frame_size=(64, 64), frame_limit=20)
y_encoded, label_encoder = encode_labels(y)

from utils.preprocessing import one_hot_encode
y_onehot = one_hot_encode(y_encoded)

X_test, y_test = X, y_onehot

# Load model
model = load_model("models/asl_cnn_lstm.h5")

# Evaluate test data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc*100:.2f}%")

# Predict classes
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_))

# Try to load training history for plotting
history_path = 'training_history.npz'
if os.path.exists(history_path):
    history = np.load(history_path)
    loss = history['loss']
    val_loss = history['val_loss']
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
    plt.title('Epoch vs Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r--', label='Validation Accuracy')
    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("epoch_loss_accuracy.png", dpi=300)   
    plt.show()
print("✅ Diagram saved as 'epoch_loss_accuracy.png'")

