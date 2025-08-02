# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os, cv2, numpy as np
# from tensorflow.keras.models import load_model
# from utils.preprocessing import encode_labels

# app = Flask(__name__)
# CORS(app)  

# model = load_model("models/asl_cnn_lstm.h5")
# classes = sorted(os.listdir("extracted_frames"))
# _, label_encoder = encode_labels(np.array(classes))

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['video']
#     file_path = "temp_video.mp4"
#     file.save(file_path)

#     cap = cv2.VideoCapture(file_path)
#     frames = []
#     while len(frames) < 20:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         resized = cv2.resize(frame, (64, 64))
#         normalized = resized.astype("float32") / 255.0
#         frames.append(normalized)
#     cap.release()
#     os.remove(file_path)

#     if len(frames) == 20:
#         X_input = np.expand_dims(np.array(frames), axis=0)
#         prediction = model.predict(X_input)
#         class_index = np.argmax(prediction)
#         predicted_word = label_encoder.inverse_transform([class_index])[0]
#         return jsonify({'predicted_word': predicted_word})
#     else:
#         return jsonify({'error': 'Insufficient frames'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)



# start with translation process



# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os, cv2, numpy as np
# from tensorflow.keras.models import load_model
# from utils.preprocessing import encode_labels
# from translations import translations  

# app = Flask(__name__)
# CORS(app)

# model = load_model("models/asl_cnn_lstm.h5")
# classes = sorted(os.listdir("extracted_frames"))
# _, label_encoder = encode_labels(np.array(classes))

# @app.route('/predict', methods=['POST'])
# def predict():
#     file = request.files['video']
#     file_path = "temp_video.mp4"
#     file.save(file_path)

#     cap = cv2.VideoCapture(file_path)
#     frames = []
#     while len(frames) < 20:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         resized = cv2.resize(frame, (64, 64))
#         normalized = resized.astype("float32") / 255.0
#         frames.append(normalized)
#     cap.release()
#     os.remove(file_path)

#     if len(frames) == 20:
#         X_input = np.expand_dims(np.array(frames), axis=0)
#         prediction = model.predict(X_input)
#         class_index = np.argmax(prediction)
#         predicted_word = label_encoder.inverse_transform([class_index])[0]

#         translation = translations.get(predicted_word.lower(), {"si": "N/A", "ta": "N/A"})

#         return jsonify({
#             'predicted_word': predicted_word,
#             'sinhala': translation["si"],
#             'tamil': translation["ta"]
#         })
#     else:
#         return jsonify({'error': 'Insufficient frames'}), 400

# if __name__ == '__main__':
#     app.run(debug=True)








# test code for webcam start
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, cv2, numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import encode_labels
from translations import translations
import base64
import tempfile

app = Flask(__name__)
CORS(app)

model = load_model("models/asl_cnn_lstm.h5")
classes = sorted(os.listdir("extracted_frames"))
_, label_encoder = encode_labels(np.array(classes))

def preprocess_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while len(frames) < 20:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (64, 64))
        normalized = resized.astype("float32") / 255.0
        frames.append(normalized)
    cap.release()
    return frames

def predict_word(frames):
    if len(frames) == 20:
        X_input = np.expand_dims(np.array(frames), axis=0)
        prediction = model.predict(X_input)
        class_index = np.argmax(prediction)
        predicted_word = label_encoder.inverse_transform([class_index])[0]
        translation = translations.get(predicted_word.lower(), {"si": "N/A", "ta": "N/A"})
        return {
            'predicted_word': predicted_word,
            'sinhala': translation["si"],
            'tamil': translation["ta"]
        }
    else:
        return {'error': 'Insufficient frames'}

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['video']
    file_path = "temp_video.mp4"
    file.save(file_path)

    frames = preprocess_video(file_path)
    os.remove(file_path)
    result = predict_word(frames)
    
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    data = request.json.get("video_base64")
    if not data:
        return jsonify({"error": "No video data"}), 400

    # Decode base64 video and save temporarily
    video_bytes = base64.b64decode(data.split(",")[1])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    frames = preprocess_video(temp_path)
    os.remove(temp_path)
    result = predict_word(frames)

    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

