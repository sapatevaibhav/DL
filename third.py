import tensorflow as tf
from tensorflow import keras
from keras import layers,models
import numpy as np

(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data();

x_test=x_test/255
x_train=x_train/255

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

#create encoder
encoder=models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    layers.MaxPooling2D(2,2),

])

#decoder
decoder=models.Sequential([
    layers.Conv2DTranspose(64,(3,3),activation="relu", input_shape=(7,7,64),padding='same'),
    layers.UpSampling2D((2,2)),
    layers.Conv2DTranspose(32,(3,3),activation='relu',padding='same'),
    layers.UpSampling2D((2,2)),
    layers.Conv2DTranspose(1,(3,3),activation='sigmoid',padding='same')
])

#crete autoencoder
autoencoder=models.Sequential([encoder,decoder])

#compile
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#train
h=autoencoder.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=2,batch_size=128)

# Anomaly detection: Compare reconstruction error to detect anomalies
decoded_imgs = autoencoder.predict(x_test)
reconstruction_error = np.mean(np.abs(decoded_imgs - x_test), axis=(1, 2, 3))
threshold = np.percentile(reconstruction_error, 95)  # Set threshold for anomaly detection
anomalies = reconstruction_error > threshold

print(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} test samples.")
