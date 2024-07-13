import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data loading and preprocessing
def load_images_and_labels(image_dir, label_file, image_size=(256, 256)):
    images = []
    labels = []
    with open(label_file, 'r') as file:
        for line in file:
            filename, label = line.strip().split(',')
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, image_size)
            images.append(img)
            labels.append(float(label))
    return np.array(images), np.array(labels)

# Define the CNN model with Batch Normalization
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3)),
        BatchNormalization(),
        Activation('relu'),
        Flatten(),
        Dense(128),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Data augmentation generator
def create_data_generator():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# Load and preprocess data
image_dir = 'path/to/your/images'
label_file = 'path/to/your/labels.txt'
images, labels = load_images_and_labels(image_dir, label_file)

# Split data
from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize and train model
model = create_model((256, 256, 3))
train_datagen = create_data_generator()
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)

history = model.fit(train_generator, epochs=20, validation_data=(val_images, val_labels))

# Evaluate the model
test_images, test_labels = load_images_and_labels('path/to/test/images', 'path/to/test/labels.txt')
model.evaluate(test_images, test_labels)

# Save the model if needed
model.save('dbh_model.h5')
