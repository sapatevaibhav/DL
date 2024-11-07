import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_test = X_train.reshape(-1, 784) / 255.0, X_test.reshape(-1, 784) / 255.0
Y_train, Y_test = to_categorical(Y_train, 10), to_categorical(Y_test, 10)
# Define the model
model = Sequential([
Dense(128, input_shape=(784,), activation='sigmoid'),
Dense(64, activation='sigmoid'),
Dense(10, activation='softmax')
])
# Compile and train the model
model.compile(optimizer=SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=128)

# Evaluate and plot
print(f"Test accuracy: {model.evaluate(X_test, Y_test)[1]:.4f}")
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()
#making the predictin
predictions=model.predict(X_test,batch_size=128)
