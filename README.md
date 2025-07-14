# Image Classification with Teachable Machine & Keras

This project demonstrates how to create an image recognition model using **Teachable Machine by Google**, convert it to **Keras format**, and use it in a Python script to classify input images.

##  Task Overview

1. Train an image recognition model using **Teachable Machine**.
   - Use **at least two image classes** (e.g., cats vs. dogs, apples vs. bananas).
   - Evaluate the model.

2. Download the trained model in **TensorFlow → Keras format** (`.h5` file).

3. Write a **Python script** that:
   - Loads the Keras model.
   - Accepts an input image.
   - Predicts and prints the **class** of the image.

4. Submit the following:
   - ✅ Python script (`predict.py`)
   - ✅ Exported model files (`keras_model.h5`, `labels.txt`)
   - ✅ A screenshot of the output
   - ✅ This `README.md` file

##  How the Model Was Trained

The model was trained using [Teachable Machine](https://teachablemachine.withgoogle.com/):

## python script is:

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/content/keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("16.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)




