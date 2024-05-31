import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import mediapipe as mp 
import time
import os 

DATA_PATH = 'D:\\Learning\\HeThongNhung\\NUMBER_GESTURE\\Project_Data'
ACTIONS = ['forward', 'left', 'right','backward']
video_num = 120
frame_per_vid = 5
camera = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils  

def collect_data() :
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic :
        for action in ACTIONS :
            for video in range(video_num) :
                for frame_num in range(frame_per_vid) :
                    _, frame = camera.read()
                    image, result  = mediapipe_detection(frame, holistic)
                    draw_styled_landmark(image,result)
                    cv2.putText(image, f'Action: {action.upper()} -- Video: {video}',(15,25),cv2.FONT_HERSHEY_COMPLEX,1, (255,0,0),4,cv2.LINE_AA)
                    if frame_num == 0 :
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,255),4,cv2.LINE_AA)
                        cv2.waitKey(1000)
                    
                    cv2.imshow('Collecting Data Frame', image)
                    key_points = extract_keypoints(result)
                    numpy_path = os.path.join(DATA_PATH,action,str(video),str(frame_num))
                    np.save(numpy_path,key_points)
                    if cv2.waitKey(1) & 0xFF == ord('q') :
                        break
   
        camera.release()
        cv2.destroyAllWindows()

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result


def draw_styled_landmark(image, result) :
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 2),
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 1))
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 2),
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 1))

def extract_keypoints(result):
    right_hand_points = np.zeros(21*3)
    if result.right_hand_landmarks:
        right_hand_points = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
    return right_hand_points

collect_data()