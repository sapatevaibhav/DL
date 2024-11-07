import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model

# Load CIFAR-10 dataset and preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = tf.image.resize(x_train / 255.0, (64, 64)), tf.image.resize(x_test / 255.0, (64, 64))

# Load pre-trained VGG16 model, freeze base layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
for layer in base_model.layers: layer.trainable = False

# Add custom classifier
x = Flatten()(base_model.output)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax')(x)

# Create and compile model
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and evaluate model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)
test_acc = model.evaluate(x_test, y_test, verbose=0)[1]

# Predict and visualize results
predictions = model.predict(x_test)
plt.imshow(x_test[10])
plt.title(f"Pred: {predictions[10].argmax()}, Actual: {y_test[10][0]}")
plt.show()

print(f'Test accuracy: {test_acc * 100:.2f}%')
