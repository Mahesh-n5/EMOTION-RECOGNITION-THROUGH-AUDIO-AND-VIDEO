# Emotion Recognition from Audio & Video 🎭🔊

## 📌 Overview
A deep learning system that detects 7 human emotions (Happy 😊, Sad 😢, Fear 😨, Angry 😠, Surprise 😲, Disgust 🤢, Neutral 😐) from both **speech audio** and **facial video** inputs. The CNN model was trained on the RAVDESS dataset achieving **85% accuracy**.
## 🛠️ Technical Details
- **Model Architecture**: Custom CNN with dual input branches (audio + video)
- **Core Files**:
  - `model.json` - Model architecture
  - `model_weights.h5` - Video weights
  - `audio_weights.h5` - Audio weights
- **Features Extracted**:
  - Audio: MFCC coefficients
  - Video: Facial landmarks using OpenCV
- **Output**: Real-time emotion probability bars (0-100%)
## 🚀 Run Flask server
  - python app.py -
  -- Open http://localhost:5000 in your browser
