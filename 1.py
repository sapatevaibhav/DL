# Import necessary packages
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random
# Load the MNIST dataset (handwritten digits)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Normalize the data to range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0
# Define the network architecture
# Sequential model with layers specified in sequence
model = keras.Sequential([
keras.layers.Flatten(input_shape=(28, 28)), # Flatten layer to reshape input to a 1D array
keras.layers.Dense(128, activation="relu"), # Dense layer with 128 units and ReLU activation
keras.layers.Dense(10, activation="softmax") # Output layer with 10 units for 10 classes (0-9)
])
# Display the model architecture summary
model.summary()
# Compile the model with SGD optimizer and sparse categorical crossentropy loss
model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy",
metrics=['accuracy'])
# Train the model on training data with validation on test data for 3 epochs
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3)
# Evaluate the model on test data and print test loss and accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Loss = %.3f" % test_loss)
print("Accuracy = %.3f" % test_acc)
# Select a random image from the test set
n = random.randint(0, len(x_test) - 1)
# Display the randomly selected test image
plt.imshow(x_test[n], cmap='gray')
plt.title(f"Actual Label: {y_test[n]}")
plt.show()
# Predict the class probabilities for the entire test set
predicted_values = model.predict(x_test)
# Display the selected test image again
plt.imshow(x_test[n], cmap='gray')
plt.show()
# Print the predicted class for the selected test image
print('Predicted Value:', predicted_values[n].argmax())
# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.title("Training and Validation Accuracy")
plt.show()
# Plot training and validation loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')


plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.title("Training and Validation Loss")
plt.show()