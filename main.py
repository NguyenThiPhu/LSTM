import cv2 
import matplotlib.pyplot as plt
import numpy as np 
import mediapipe as mp 
import time
import os 
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from PIL import Image



DATA_PATH = 'C:\Project_LSTM\Project_Data'
ACTIONS = ['hello', 'love', 'like', 'unlike']
video_num = 30
frame_per_vid = 30
camera = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils  

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir = log_dir)

st.title('Project: Using LSTM Network to recognize people\'s gesture')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    result = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, result

def draw_landmark(image, result) :
    mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def draw_styled_landmark(image, result) :
    mp_drawing.draw_landmarks(image, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                            mp_drawing.DrawingSpec(color = (0,0,255),thickness = 2, circle_radius = 2),
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 1))
    mp_drawing.draw_landmarks(image, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 2),
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 1))
    mp_drawing.draw_landmarks(image, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 2),
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 1))
    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 2),
                            mp_drawing.DrawingSpec(color = (0,130,48),thickness = 2, circle_radius = 1))

def extract_keypoints(result) :
    face_points = []
    pose_points = []
    right_hand_points = []
    left_hand_points = []
    if result.face_landmarks is not None:
        for res in result.face_landmarks.landmark :
            temp = [res.x, res.y, res.z]
            face_points.append(temp)
    else :
        face_points = np.zeros(468*3)
     
    if result.pose_landmarks is not None:
        for res in result.pose_landmarks.landmark :
            temp = [res.x, res.y, res.z, res.visibility]
            pose_points.append(temp)
    else :
        pose_points = np.zeros(33*4)
 
    if result.left_hand_landmarks is not None:
        for res in result.left_hand_landmarks.landmark :
            temp = [res.x, res.y, res.z]
            left_hand_points.append(temp)
    else :
        left_hand_points = np.zeros(21*3)
   
    if result.right_hand_landmarks is not None:   
        for res in result.right_hand_landmarks.landmark :
            temp = [res.x, res.y, res.z]
            right_hand_points.append(temp)
    else :
        right_hand_points = np.zeros(21*3)
    face_points = np.array(face_points).flatten()
    pose_points = np.array(pose_points).flatten()
    right_hand_points = np.array(right_hand_points).flatten()
    left_hand_points = np.array(left_hand_points).flatten()
    return np.concatenate([pose_points, face_points, left_hand_points, right_hand_points])

def preprocessing () :
    label_mapping = {label:index for index, label in enumerate(ACTIONS)}
    sequences = []
    labels = []

    for action in ACTIONS :
        for video in range(video_num) :
            window = []
            for frame in range(frame_per_vid) :
                temp = np.load(os.path.join(DATA_PATH, action, str(video),f'{frame}.npy'))
                window.append(temp)

            sequences.append(window)
            labels.append(label_mapping[action])

    x = np.array(sequences)
    y = np.array(labels)


    y = to_categorical(y).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
    return x_train, x_test, y_train, y_test


def model_building(x_train,x_test,y_train,y_test) :
    model = Sequential([
        tf.keras.layers.Input((30, 1662)),
        tf.keras.layers.LSTM(64, return_sequences = True, activation = 'relu'),
        tf.keras.layers.LSTM(128, return_sequences = True, activation = 'relu'),
        tf.keras.layers.LSTM(64, return_sequences = False, activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(4, activation = 'softmax')]
    )

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
    model.fit(x_train,y_train, epochs = 1000, callbacks = [tb_callback])

    return model 
    
def main () : 
    # x_train, x_test, y_train, y_test = preprocessing ()
    # model = model_building(x_train, x_test, y_train, y_test)
    # model.save('action.h5')
    loaded_model = load_model('action.h5')

    sequence = []

    threshold = 0.5
    st.title("Webcam Live Feed")
    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic :
        while run:
            _, frame = camera.read()
            
            image, result  = mediapipe_detection(frame, holistic)
            draw_styled_landmark(image,result)
            
            keypoints = extract_keypoints(result)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = loaded_model.predict(np.expand_dims(sequence, axis=0))[0]
                    
                cv2.putText(image,f'Predicted: {ACTIONS[np.argmax(res)].upper()}' , (100,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                #st.write(f'Predicted: {ACTIONS[np.argmax(res)].upper()}')
            FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    camera.release()

if __name__ == '__main__':
    main()

