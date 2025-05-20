import cv2
import keras
import numpy as np
import time

model_from_json = keras.models.model_from_json
emotion_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

with open(r"model\\model.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights(r"model\\model.weights.h5")
print("Loaded model from disk")

cap = None
camera_index = 0  

while cap is None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Camera {camera_index} not found. Trying the next index.")
        camera_index += 1
        if camera_index > 5: 
            print("No camera found. Exiting...")
            exit()

emotion_probabilities = []
start_time = time.time()

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(frame, (1280, 720))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    emotion_sum = np.zeros(len(emotion_dict))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)

        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)[0]
        emotion_sum += emotion_prediction  

        maxindex = int(np.argmax(emotion_prediction))
        emotion = emotion_dict[maxindex]

        cv2.putText(frame, emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if len(faces) > 0:
        emotion_probabilities.append(emotion_sum / np.sum(emotion_sum))

    cv2.imshow('Emotion Detection', frame)

    current_time = time.time()
    if current_time - last_save_time >= 1.0:  
        if emotion_probabilities:
            with open("detected_emotions.txt", "w") as f:
                cumulative_emotions = {emotion: 0.0 for emotion in emotion_dict.values()}
                
                for probabilities in emotion_probabilities:
                    for index, probability in enumerate(probabilities):
                        emotion_name = emotion_dict[index]
                        cumulative_emotions[emotion_name] += probability
                
                total_count = len(emotion_probabilities)
                
                for emotion, total_probability in cumulative_emotions.items():
                    average_probability = total_probability / total_count if total_count > 0 else 0.0
                    f.write(f"{emotion}: {average_probability:.2f}\n")  
            
            last_save_time = current_time  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Detected emotions have been updated in detected_emotions.txt every second.")
