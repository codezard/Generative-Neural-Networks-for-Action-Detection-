from datacreation import mediapipe_detection,draw_landmarks, extractKeypoints
import sklearn
import tensorflow
import numpy as np
import os
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
"""
The data will be taken from the data set and preprocessed for the model.
The model is all created and preprocessed data is used to train the model.

"""

# Data Retrieval
actions = np.array(['Hello','Thank You','I Love You'])
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
no_sequences=30
sequence_length=30
data_path = os.path.join("/".join(os.getcwd().split('\\')),'Data')
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(data_path,action,str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Converting data from list to numpy array
y = to_categorical(labels).astype(int)
X = np.array(sequences)

# Dividing features and labels into training and testing variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print('Training Data Shape')
print('X: {} and Y(Labels): {}'.format(X_train.shape,y_train.shape))
print()
print('Testing Data Shape')
print('X: {} and Y(Labels): {}'.format(X_test.shape,y_test.shape))

# Model Creation
model = tensorflow.keras.models.Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])
# Train the model for 500 epochs
model.fit(X_train, y_train, epochs = 500)
# The model is stored by the name of action.h5
model.save('action.h5')
# model.load_weights('action.h5')
