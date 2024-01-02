import cv2
import numpy as np
import tensorflow as tf

# Load the DeepLabV3 model pre-trained on the COCO dataset
model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
model.trainable = False

# Load an image (replace 'your_image_path.jpg' with the actual path to your image)
image_path = '100.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize the image to the size expected by the model
input_size = (224, 224)
image_resized = cv2.resize(image, input_size)

# Normalize the image
image_resized = tf.keras.applications.densenet.preprocess_input(image_resized)

# Add an extra dimension to match the model's expected input shape
image_resized = np.expand_dims(image_resized, axis=0)

# Get the predicted mask from the model
mask = model.predict(image_resized)

# Convert the mask to binary format
mask_binary = (mask > 0.5).astype(np.uint8)

# Apply the mask to the original image
image_segmented = image * mask_binary

# Save or display the result
cv2.imwrite('output_image.jpg', cv2.cvtColor(image_segmented[0], cv2.COLOR_RGB2BGR))
