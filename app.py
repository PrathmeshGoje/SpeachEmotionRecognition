from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import os
import librosa
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('mymodel.h5')
print("Model loaded.")


@app.route('/')
def home():
    return render_template('index.html')


def extract_mfcc(wav_file):
    # Extract MFCC features from the audio file
    y, sr = librosa.load(wav_file)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs


def predict(model, wav_file):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_file)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Define the path to the static folder
            static_folder = os.path.join(app.root_path, 'static')
            if not os.path.exists(static_folder):
                os.makedirs(static_folder)
            # Define file path
            file_path = os.path.join(static_folder, file.filename)
            file.save(file_path)
            try:
                emotion = predict(model, file_path)
            except Exception as e:
                return f"Error during prediction: {str(e)}"
            finally:
                # Clean up the file after processing
                if os.path.exists(file_path):
                    os.remove(file_path)
            return render_template('index.html', emotion=emotion, audio_file=file_path)

    return render_template('index.html', emotion=None, audio_file=None)


if __name__ == '__main__':
    app.run()
