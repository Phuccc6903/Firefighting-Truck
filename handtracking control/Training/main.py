import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_PATH = '\\Project_Data'
ACTIONS = ['forward', 'left', 'right', 'backward']
video_num = 120
frame_per_vid = 5
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

def draw_landmark(image, result):
    mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def draw_styled_landmark(image, result):
    mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                              mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=1))
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,130,48), thickness=2, circle_radius=1))

def extract_keypoints(result):
    right_hand_points = np.zeros(21*3)
    if result.right_hand_landmarks:
        right_hand_points = np.array([[res.x, res.y, res.z] for res in result.right_hand_landmarks.landmark]).flatten()
    return right_hand_points

def preprocessing():
    label_mapping = {label: index for index, label in enumerate(ACTIONS)}
    sequences = []
    labels = []
    for action in ACTIONS:
        for video in range(video_num):
            window = []
            for frame in range(frame_per_vid):
                temp = np.load(os.path.join(DATA_PATH, action, str(video), f'{frame}.npy'))
                window.append(temp)
            sequences.append(window)
            labels.append(label_mapping[action])
    x = np.array(sequences)
    y = np.array(labels)
    y = to_categorical(y).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    return x_train, x_test, y_train, y_test

def model_building(x_train, x_test, y_train, y_test):
    model = Sequential([
        tf.keras.layers.Input((9, 21*3)),
        tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'),
        tf.keras.layers.LSTM(256, return_sequences=True, activation='relu'),
        tf.keras.layers.LSTM(128, return_sequences=False, activation='relu'),
        tf.keras.layers.Dense(2048, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(len(ACTIONS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(x_train, y_train, epochs=150)
    return model

def main():
    x_train, x_test, y_train, y_test = preprocessing()
    model = model_building(x_train, x_test, y_train, y_test)
    model.save('modeling.h5')

if __name__ == '__main__':
    main()
