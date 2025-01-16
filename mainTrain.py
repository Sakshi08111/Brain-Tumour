import cv2
import os
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical

# Update dataset path
dataset_path = r'C:\Users\A2Z\Desktop\brain tumour detection\dataset'

# Create paths for 'no' and 'yes' subdirectories
no_tumor_path = os.path.join(dataset_path, 'no')
yes_tumor_path = os.path.join(dataset_path, 'yes')

# List of images in 'no' and 'yes' subdirectories
no_tumor_images = os.listdir(no_tumor_path)
yes_tumor_images = os.listdir(yes_tumor_path)

dataset = []
label = []


INPUT_SIZE = 64

# Load images from 'no' subdirectory
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[-1].lower() == 'jpg':
        image = cv2.imread(os.path.join(no_tumor_path, image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

# Load images from 'yes' subdirectory
for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[-1].lower() == 'jpg':
        image = cv2.imread(os.path.join(yes_tumor_path, image_name))
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

# Convert to NumPy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize pixel values
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Model Building
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Modify the last Dense layer to match the number of classes and use 'softmax' activation
model.add(Dense(2))  # or model.add(Dense(1)) for binary classification with sigmoid activation
model.add(Activation('softmax'))  # or model.add(Activation('sigmoid'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)

# Save the model
model.save('BrainTumor10EpochsCategorical', save_format='tf')

