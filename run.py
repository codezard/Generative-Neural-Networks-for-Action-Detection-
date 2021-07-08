# from datacreation import mediapipe_detection,draw_landmarks, extractKeypoints
import cv2
import sklearn
import tensorflow
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM, Dense
import mediapipe as mp

mp_holistic = mp.solutions.holistic # Holistic Model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image, model):
    """ 
    Convert the bgr to rgb and set it to unwritable
    Process it through the model
    Set it back to writable and convert it back from rbg to bgr
    """ 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # Color Conversion - BGR to RGB
    image.flags.writeable = False
    results = model.process(image) # Processing image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Color Conversion - RGB to BGR
    return image, results
def draw_landmarks(image,results):
    # Draw Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACE_CONNECTIONS,
                             landmark_drawing_spec = mp_drawing.DrawingSpec(color = (80,110,10), thickness = 1, circle_radius = 1),
                             connection_drawing_spec = mp_drawing.DrawingSpec(color = (80,256,121), thickness = 1, circle_radius = 1)
                             )
    # Draw Pose Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                             landmark_drawing_spec = mp_drawing.DrawingSpec(color = (80,22,10), thickness = 2, circle_radius = 2),
                             connection_drawing_spec = mp_drawing.DrawingSpec(color = (80,44,121), thickness = 2, circle_radius = 2)
                             )
    # Draw Left Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             landmark_drawing_spec = mp_drawing.DrawingSpec(color = (121,22,76), thickness = 4, circle_radius = 2),
                             connection_drawing_spec = mp_drawing.DrawingSpec(color = (121,44,250), thickness = 4, circle_radius=2)
                             )
    # Draw Right Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                             landmark_drawing_spec = mp_drawing.DrawingSpec(color = (121,22,76), thickness = 4, circle_radius = 2),
                             connection_drawing_spec = mp_drawing.DrawingSpec(color = (121,44,250), thickness = 4, circle_radius =2)
                             )
def extractKeypoints(results):
    rightHand = np.zeros(21*3)
    leftHand = np.zeros(21*3)
    pose = np.zeros(33*3)
    face = np.zeros(468*3)
    # for res in results.right_hand_landmarks.landmark:
    #     test = np.array([res.x,res.y,res.z])
    #     rightHand.append(test)
    if results.right_hand_landmarks:
        rightHand = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()

    if results.left_hand_landmarks:
        leftHand = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten()

    if results.pose_landmarks:
        pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()

    if results.face_landmarks:
        face = np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten()

    return np.concatenate([pose, face, leftHand, rightHand])

actions = np.array(['Hello','Thank You','I Love You'])
# Model Creation
model = tensorflow.keras.models.Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
model.load_weights('action.h5')

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)
        
        # Draw landmarks
        draw_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = extractKeypoints(results)
#         sequence.insert(0,keypoints)
#         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            
            
        #3. Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            # image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        