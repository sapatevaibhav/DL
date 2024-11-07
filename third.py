import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Step b: Load and preprocess the dataset (MNIST for simplicity)
(x_train, _), (x_test, _) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))  # Reshape for CNNs
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# Step c: Build the Encoder-Decoder (Autoencoder) Model
input_img = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
decoded = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(decoded)

# Model
autoencoder = models.Model(input_img, decoded)

# Step e: Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step d: Train the model and capture the training history
history = autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))

# Anomaly detection: Compare reconstruction error to detect anomalies
decoded_imgs = autoencoder.predict(x_test)
reconstruction_error = np.mean(np.abs(decoded_imgs - x_test), axis=(1, 2, 3))
threshold = np.percentile(reconstruction_error, 95)  # Set threshold for anomaly detection
anomalies = reconstruction_error > threshold

print(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} test samples.")

# Plot the loss curves for training and validation
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
