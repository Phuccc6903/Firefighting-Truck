from flask import Flask, render_template, Response
import cv2
import matplotlib.pyplot as plt
import numpy as np 
import mediapipe as mp 
import time

from keras.models import load_model
from PIL import Image

ACTIONS = ['forward', 'left', 'right', 'backward']

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

def draw_styled_landmark(image, result):
    if result.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=1))
        
def extract_keypoints(result):
    right_hand_points = np.zeros(21*3)
    if result.right_hand_landmarks:
        right_hand_points = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
    return right_hand_points

app = Flask(__name__)
global predicted_action, hand_detected

predicted_action = None
hand_detected = False

camera = cv2.VideoCapture(0)


if not camera.isOpened():
    raise RuntimeError("Cannot open camera")

def gen_frames():
    global predicted_action,hand_detected
    model = load_model('modeling.h5')


    sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            image, result = mediapipe_detection(frame, holistic)
            draw_styled_landmark(image, result)
            keypoints = extract_keypoints(result)
            sequence.append(keypoints)
            sequence = sequence[-5:]

            hand_detected = result.right_hand_landmarks is not None

            if len(sequence) == 5 and hand_detected:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = ACTIONS[np.argmax(res)]
                print(predicted_action)
                cv2.putText(image, f'Predicted: {predicted_action.upper()}', (100, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            
            

            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                print("Failed to encode frame")
                break
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                camera.release()
                break
            cv2.imshow('Frame', frame)

            hand_detected=False
    hand_detected=False


@app.route('/')
def index():
    # Hiển thị trang chủ
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Trả về luồng video
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def send_prediction():
    global predicted_action, hand_detected
    if predicted_action is not None and hand_detected == True:
        print(predicted_action)
        return Response(predicted_action, content_type= 'text/plain')
    else:
        return Response(None)


if __name__ == '__main__':
    # Chạy ứng dụng Flask trên tất cả các địa chỉ IP trên cổng 5000
    app.run(host='0.0.0.0', port=5000)


