#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# In[19]:


train_images_path = "C:/Users/Sneha Nishani/Desktop/mini/Arecanut_dataset/Arecanut_dataset/train"
test_images_path = "C:/Users/Sneha Nishani/Desktop/mini/Arecanut_dataset/Arecanut_dataset/test"
train_csv_path = "C:/Users/Sneha Nishani/Desktop/mini/Arecanut_dataset/Arecanut_dataset/train/trainnewdata.csv"
test_csv_path="C:/Users/Sneha Nishani/Desktop/mini/Arecanut_dataset/Arecanut_dataset/test/testdata.csv"


# In[20]:


train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)


# In[21]:


print(train_df)


# In[22]:


print(test_df)


# In[23]:


activities = train_df['label'].unique().tolist()
print("Activities:", activities)


# In[24]:


# Define performance metrics
performance_metrics = ["accuracy", "precision", "recall", "f1-score"]


# In[25]:


# Map activity labels to integer indices
label_to_index = {label: index for index, label in enumerate(activities)}
print(label_to_index)


# In[26]:


def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return image


# In[11]:


def load_dataset(df, images_path, label_to_index=None):
    images = []
    labels = []
    for _, row in df.iterrows():
        image_path = os.path.join(images_path, row['filename'])
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at path: {image_path}")
            continue
        try:
            image = load_and_preprocess_image(image_path)
            images.append(image)
            if label_to_index is not None:
                label = label_to_index[row['label']]
                labels.append(label)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    if label_to_index is not None:
        return np.array(images), np.array(labels)
    else:
        return np.array(images)


# In[12]:


#Load and preprocess datasets
X_train, y_train = load_dataset(train_df, train_images_path, label_to_index)
X_test = load_dataset(test_df, test_images_path)

# Ensure labels are integers for training
y_train = np.array(y_train, dtype=int)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=len(activities))


# In[13]:


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)


# In[14]:


model = Sequential()

# First convolutional block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Second convolutional block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Third convolutional block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Fourth convolutional block
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Flattening the network and adding fully connected layers
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(len(activities), activation='softmax'))


# In[15]:


# Model compilation
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# In[16]:


# Model training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)


# In[28]:


def plot_training_history(history):
    # Accuracy plot
    plt1.figure(figsize=(14, 5))

    # Accuracy subplot
    plt1.subplot(1, 2, 1)
    plt1.plot(history.history['accuracy'], label='Training Accuracy')
    plt1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt1.title('Training and Validation Accuracy')
    plt1.xlabel('Epoch')
    plt1.ylabel('Accuracy')
    plt1.legend()

    # Loss subplot
    plt1.subplot(1, 2, 2)
    plt1.plot(history.history['loss'], label='Training Loss')
    plt1.plot(history.history['val_loss'], label='Validation Loss')
    plt1.title('Training and Validation Loss')
    plt1.xlabel('Epoch')
    plt1.ylabel('Loss')
    plt1.legend()

    plt1.show()

# Plot the training history
plot_training_history(history)


# In[2]:


import time
import collections
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import tkinter as tk


# In[ ]:


stop_capture = False

def capture_and_predict():
    global stop_capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    buffer_size = 90  # 5 seconds buffer assuming 30 FPS
    prediction_buffer = collections.deque(maxlen=buffer_size)
    start_time = time.time()
    output_text = ''
    activity_index = 0

    while True:
        if stop_capture:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)

        # Model inference
        prediction = model.predict(frame_expanded)
        predicted_activity = activities[np.argmax(prediction)]
        prediction_buffer.append(predicted_activity)

        # Switch activity every 5 seconds
        if time.time() - start_time >= 3:
            # Get the most frequent activity in the buffer
            if prediction_buffer:
                most_common_activity = collections.Counter(prediction_buffer).most_common(1)[0][0]
                output_text = most_common_activity

            # Clear the buffer and reset the timer
            prediction_buffer.clear()
            start_time = time.time()

            # Switch to the next activity
            activity_index = (activity_index + 1) % len(activities)

        # Output visualization
        cv2.putText(frame, f'Activity: {output_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Activity Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_recognition():
    global stop_capture
    stop_capture = False
    recognition_thread = threading.Thread(target=capture_and_predict)
    recognition_thread.start()

def stop_recognition():
    global stop_capture
    stop_capture = True

# Tkinter GUI setup
root = tk.Tk()
root.title("Activity Recognition Control")

start_button = tk.Button(root, text="Start Recognition", command=start_recognition)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Recognition", command=stop_recognition)
stop_button.pack(pady=10)

root.mainloop()


# In[1]:


stop_capture = False

def capture_and_predict():
    global stop_capture
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    buffer_size = 150  # 5 seconds buffer assuming 30 FPS
    prediction_buffer = collections.deque(maxlen=buffer_size)
    start_time = time.time()
    output_text = ''
    activity_index = 0

    while True:
        if stop_capture:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_resized = cv2.resize(frame, (64, 64))
        frame_normalized = frame_resized / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)

        # Model inference
        prediction = model.predict(frame_expanded)
        predicted_activity = activities[np.argmax(prediction)]
        prediction_buffer.append(predicted_activity)

        # Switch activity every 5 seconds
        if time.time() - start_time >= 3:
            # Get the most frequent activity in the buffer
            most_common_activity = collections.Counter(prediction_buffer).most_common(1)[0][0]
            output_text = most_common_activity

            # Clear the buffer and reset the timer
            prediction_buffer.clear()
            start_time = time.time()

            # Switch to the next activity
            activity_index = (activity_index + 1) % len(activities)

        # Output visualization
        cv2.putText(frame, f'Activity: {output_text}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Activity Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def start_recognition():
    global stop_capture
    stop_capture = False
    recognition_thread = threading.Thread(target=capture_and_predict)
    recognition_thread.start()

def stop_recognition():
    global stop_capture
    stop_capture = True

# Tkinter GUI setup
root = tk.Tk()
root.title("Activity Recognition Control")

start_button = tk.Button(root, text="Start Recognition", command=start_recognition)
start_button.pack(pady=10)

# Corrected line with closing parenthesis
stop_button = tk.Button(root, text="Stop Recognition", command=stop_recognition)
stop_button.pack(pady=10)

root.mainloop()


# In[ ]:




