import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load images and corresponding DBH measurements
def load_images_and_dbh(image_dir, label_file, image_size=(256, 256)):
    images = []
    dbh_measurements = []
    with open(label_file, 'r') as file:
        for line in file:
            filename, dbh = line.strip().split(',')
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, image_size)
                images.append(img)
                dbh_measurements.append(float(dbh))
    return np.array(images), np.array(dbh_measurements)

# Create the CNN model
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Data augmentation configuration
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

# Main execution block
if __name__ == "__main__":
    # Directory and file configuration
    image_dir = 'path/to/your/images'
    label_file = 'path/to/your/labels.txt'
    
    # Load and preprocess the dataset
    images, dbh_measurements = load_images_and_dbh(image_dir, label_file)
    
    # Split the data into training and validation sets
    train_images, val_images, train_dbh, val_dbh = train_test_split(images, dbh_measurements, test_size=0.2, random_state=42)
    
    # Create the model
    model = create_model(train_images.shape[1:])
    
    # Prepare the data augmentation generator
    train_datagen = create_data_generator()
    train_generator = train_datagen.flow(train_images, train_dbh, batch_size=32)
    
    # Train the model
    history = model.fit(train_generator, epochs=20, validation_data=(val_images, val_dbh))
    
    # Optionally save the model
    model.save('dbh_prediction_model.h5')

    # Print model summary
    model.summary()
