from flask import Flask, render_template, jsonify, request, Response
import cv2
import keras
import numpy as np
import threading
import subprocess
import librosa
import tensorflow as tf
import noisereduce as nr
import sounddevice as sd

app = Flask(__name__)

test_process = None

emotion_data_video = {}
emotion_data_audio = {} 

model_from_json = keras.models.model_from_json
emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

with open(r"model\\model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model_video = model_from_json(loaded_model_json)
emotion_model_video.load_weights(r"model\\model.weights.h5")

model_audio = tf.keras.models.load_model('final_audio_emotion_recognition_model.h5')
class_labels_audio = ['Angry', 'Happy', 'Sad', 'Neutral', 'Fearful', 'Disgust', 'Surprised', 'Calm']

cap = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def update_emotion_data_video(emotion_prediction):
    global emotion_data_video
    emotion_data_video = {emotion_dict[i]: confidence * 1 for i, confidence in enumerate(emotion_prediction)}

def record_audio(duration=10, sample_rate=22050):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    audio_data = np.squeeze(audio_data)  
    return audio_data

def process_audio(audio_data, sample_rate=22050, n_mels=128):
    reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
    mel_spectrogram = librosa.feature.melspectrogram(y=reduced_noise_audio, sr=sample_rate, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

def update_audio_emotion_data():
    global emotion_data_audio
    while True:
        audio_data = record_audio(duration=2) 
        mel_spectrogram = process_audio(audio_data)
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        audio_prediction = model_audio.predict(mel_spectrogram)
        emotion_data_audio = {class_labels_audio[i]: confidence for i, confidence in enumerate(audio_prediction[0])}

audio_thread = threading.Thread(target=update_audio_emotion_data)
audio_thread.daemon = True
audio_thread.start()

def generate_frames():
    global emotion_data_video
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))  
            roi_gray = roi_gray.astype('float32') / 255  
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1) 

            emotion_prediction = emotion_model_video.predict(roi_gray)
            update_emotion_data_video(emotion_prediction[0])  
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', emotion_data_video=emotion_data_video, emotion_data_audio=emotion_data_audio)

@app.route('/emotion_data_video')
def get_emotion_data_video():
    return jsonify(emotion_data_video)

@app.route('/emotion_data_audio')
def get_emotion_data_audio():
    return jsonify(emotion_data_audio)

@app.route('/start_test', methods=['POST'])
def start_test():
    global test_process
    if test_process is None:  
        test_process = subprocess.Popen(['python', 'test.py'])
        return {'status': 'Test started'}
    return {'status': 'Test is already running'}

@app.route('/stop_test', methods=['POST'])
def stop_test():
    global test_process
    if test_process is not None: 
        test_process.terminate()  
        test_process = None  
        return {'status': 'Test stopped'}
    return {'status': 'No test is running'}

if __name__ == '__main__':
    app.run(debug=True)
