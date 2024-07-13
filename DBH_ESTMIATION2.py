import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from skimage.filters import gaussian

# Enhanced image loading and preprocessing
def load_and_process_images(image_dir, label_file, target_size=(224, 224)):
    images = []
    labels = []
    for line in open(label_file, 'r'):
        filename, dbh = line.strip().split(',')
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = preprocess_image(img)  # Custom preprocessing
            images.append(img)
            labels.append(float(dbh))
    return np.array(images), np.array(labels)

# Custom image preprocessing
def preprocess_image(image):
    image = gaussian(image, sigma=1, multichannel=True)  # Apply Gaussian Blur
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

# Model creation with fine-tuning
def create_finetuned_model(input_shape):
    base_model = MobileNetV2(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True  # Fine-tune all layers
    for layer in base_model.layers[:100]:  # Freeze the first 100 layers
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1)(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

# Main execution block
if __name__ == "__main__":
    image_dir = 'path/to/images'
    label_file = 'path/to/labels.txt'
    images, labels = load_and_process_images(image_dir, label_file)

    # Split the data
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define the model
    model = create_finetuned_model(train_images.shape[1:])

    # Enhanced data augmentation
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.3, height_shift_range=0.3, horizontal_flip=True, vertical_flip=True)
    train_generator = datagen.flow(train_images, train_labels, batch_size=32)

    # Train the model
    history = model.fit(train_generator, epochs=20, validation_data=(test_images, test_labels))

    # Save the model
    model.save('dbh_model_mobilenet_advanced.h5')
