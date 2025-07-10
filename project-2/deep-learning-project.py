# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# --------- Config ---------
MODEL_PATH = "cifar10_cnn_model.h5"  # Path to save/load trained model
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR-10 class labels

# --------- Build CNN Model ---------
def build_model():
    """
    Builds a simple Convolutional Neural Network (CNN) for image classification on CIFAR-10 dataset.
    """
    model = models.Sequential([
        # First convolution + pooling layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),

        # Second convolution + pooling layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Third convolution layer
        layers.Conv2D(64, (3, 3), activation='relu'),

        # Flatten and add Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)  # Output layer for 10 classes
    ])
    return model

# --------- Train & Save Model ---------
def train_and_save_model():
    """
    Trains the CNN model on the CIFAR-10 dataset and saves it to disk.
    Also plots training and validation accuracy curves.
    """
    print("Training CIFAR-10 model...")

    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize image pixel values to [0, 1]
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # Build and compile the model
    model = build_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Save the trained model to disk
    model.save(MODEL_PATH)
    print("Model saved as", MODEL_PATH)

    # Plot training vs. validation accuracy
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

# --------- Load Model & Predict ---------
def predict_image(image_path):
    """
    Loads the trained model and uses it to predict the class of a given image.
    Displays the prediction with confidence score.
    """
    # If the model is not available, train and save it
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()

    # Load the trained model and wrap it with a softmax for probabilities
    model = tf.keras.models.load_model(MODEL_PATH)
    prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

    # Read and preprocess the input image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ ERROR: Image not found. Check the file path.")
        return

    img_resized = cv2.resize(img, (32, 32))  # Resize to CIFAR-10 input size
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_normalized = img_rgb / 255.0  # Normalize to [0,1]
    img_input = img_normalized.reshape(1, 32, 32, 3)  # Add batch dimension

    # Make prediction
    prediction = prob_model.predict(img_input)
    pred_class = np.argmax(prediction)
    confidence = 100 * np.max(prediction)

    # Display result
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {class_names[pred_class]} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

# --------- Entry Point ---------
if __name__ == "__main__":
    # Provide the path to a custom image (must be 32x32 or will be resized)
    image_path = r"C:\Users\syedt\OneDrive\Desktop\images.jpeg"
    predict_image(image_path)
