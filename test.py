import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load images and labels
def load_data(data_directory):
    images = []
    labels = []
    class_folders = sorted(os.listdir(data_directory))
    for class_folder in class_folders:
        class_path = os.path.join(data_directory, class_folder)
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))  # Resize image to 28x28 pixels
            images.append(image)
            labels.append(class_folder)
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    return images, labels

# Load dataset (replace 'data_directory' with your dataset directory)
data_directory = 'dataset'
images, labels = load_data(data_directory)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Reshape data for CNN input (add a channel dimension for grayscale)
X_train = X_train.reshape((-1, 28, 28, 1))
X_test = X_test.reshape((-1, 28, 28, 1))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Function to preprocess and predict characters on a number plate image
def predict_characters(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (28, 28))  # Resize image to 28x28 pixels
    resized_image = resized_image.astype(np.float32) / 255.0
    resized_image = np.expand_dims(resized_image, axis=-1)  # Add batch dimension
    prediction = model.predict(np.array([resized_image]))
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.classes_[predicted_class]
    return predicted_label

# Example usage:
image_path = 'test_plate.jpg'  # Replace with your test image path
predicted_label = predict_characters(image_path, model)
print(f'Predicted character: {predicted_label}')
