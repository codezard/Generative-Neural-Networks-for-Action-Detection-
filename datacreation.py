import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import mediapipe

mp_holistic =mediapipe.solutions.holistic
mp_drawing = mediapipe.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Covert the bgr to rgb and set it to unwritable.
    Process it through the model.
    Set it back to writable and convert it back from RGB to BGR.
    """
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # Color inversion - BGR to RGB
    image.flags.writeable = False
    results = model.process(image) # Processing Image
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR) # Color inversion - RGB to BGR
    return image, results

def draw_landmarks(image,results):
    # Draw Face Connections
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACE_CONNECTIONS,
                              landmark_drawing_spec = mp_drawing.DrawingSpec(color = (80,110,10), thickness = 1, circle_radius = 1),
                              connection_drawing_spec = mp_drawing.DrawingSpec(color = (80,256,121), thickness = 1, circle_radius = 1)
                              )
    # Draw Pose Connections
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              landmark_drawing_spec = mp_drawing.DrawingSpec(color = (90,110,10), thickness = 2, circle_radius = 2),
                              connection_drawing_spec = mp_drawing.DrawingSpec(color = (80,256,121), thickness = 2, circle_radius = 2)
                              )
    # Draw Right Hand Connections
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec = mp_drawing.DrawingSpec(color = (121,22,76), thickness = 2, circle_radius = 2),
                              connection_drawing_spec = mp_drawing.DrawingSpec(color = (80,256,121), thickness = 2, circle_radius = 2)
                              )
    # Draw Left Hand Connections
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              landmark_drawing_spec = mp_drawing.DrawingSpec(color = (121,22,76), thickness = 2, circle_radius = 2),
                              connection_drawing_spec = mp_drawing.DrawingSpec(color = (80,256,121), thickness = 2, circle_radius = 2)
                              )

val=os.getcwd()
val=val.split("\\")
val = "/".join(val)
# Path for exported data, numpy arrays
data_path = os.path.join(val,'Data')

#Actions that we try to detect
actions = np.array(['Hello','Thank You','I Love You'])

# Thrity videos worth of data for each action
sequences = 30

#Each video is going to be of 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(sequences):
        os.makedirs(os.path.join(data_path,action,str(sequence)))

def extractKeypoints(results):
    """
    Extracts keypoints from the preprocessed result.
    Divides the keypoints into their designated arrays depending on the landmark.
    Concatenates the whole array in order of pose, face, leftHand and rightHand
    """ 

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

def collect_data(actions):
    cap = cv2.VideoCapture(0)
    # Accessing the webcam
    if not cap.isOpened():
        print('Cannot open the camera.')
        exit()
    #Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        # Loop through actions
        for action in actions:
            # Loop through sequences aka videos
            for sequence in range(sequences):
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):

                    # Read Feed
                    ret, frame = cap.read()

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    #Draw landmarks
                    draw_landmarks(image, results)

                    if frame_num ==1:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    if not ret:
                        print('Cannot receive frame (Stream end?). Exiting')
                        break
                    # Export keypoints
                    keypoints = extractKeypoints(results)
                    npyPath = os.path.join(data_path,action, str(sequence), str(frame_num))
                    np.save(npyPath, keypoints)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    # Release the capture, once the stream ends
    cap.release()
    cv2.destroyAllWindows()

collect_data(actions)
