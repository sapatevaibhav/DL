import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load pre-trained model from TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load an internet image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/7/70/017_Great_blue_turaco_at_Kibale_forest_National_Park_Photo_by_Giles_Laurent.jpg/500px-017_Great_blue_turaco_at_Kibale_forest_National_Park_Photo_by_Giles_Laurent.jpg"  # Example image URL
image_path = tf.keras.utils.get_file("test_image2.jpg", origin=image_url)

# Load and preprocess the image
img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(img, (320, 320))
input_tensor = tf.convert_to_tensor([resized_img], dtype=tf.uint8)

# Run detection
detections = detector(input_tensor)

# Visualize detections
plt.imshow(img)
for i in range(detections['detection_scores'][0].numpy().size):
    if detections['detection_scores'][0][i].numpy() >= 0.3:
        bbox = detections['detection_boxes'][0][i].numpy()
        y_min, x_min, y_max, x_max = bbox
        h, w, _ = img.shape
        plt.gca().add_patch(plt.Rectangle((x_min * w, y_min * h), (x_max - x_min) * w, (y_max - y_min) * h,
                                          edgecolor='red', fill=False, linewidth=2))
plt.axis("off")
plt.show()
