import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load and preprocess images
def load_images_and_labels(image_dir, label_file, target_size=(224, 224)):
    images = []
    labels = []
    for line in open(label_file, 'r'):
        filename, dbh = line.strip().split(',')
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = preprocess_input(img)  # Preprocess based on the MobileNetV2 requirements
            images.append(img)
            labels.append(float(dbh))
    return np.array(images), np.array(labels)

# Create a model with MobileNetV2 as the base
def create_model(input_shape, num_classes):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False  # Freeze the convolutional base
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)  # Change this based on your output needs
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Main routine
if __name__ == "__main__":
    image_dir = 'path/to/images'
    label_file = 'path/to/labels.txt'
    images, labels = load_images_and_labels(image_dir, label_file)

    # Split the data
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define the model
    model = create_model(train_images.shape[1:], 1)

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    train_generator = datagen.flow(train_images, train_labels, batch_size=32)

    # Train the model
    history = model.fit(train_generator, epochs=10, validation_data=(test_images, test_labels))

    # Save the model
    model.save('dbh_model_mobilenet.h5')
