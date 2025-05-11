import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

#loading model
model = load_model('scripts/best_model.h5')

#class names
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

#path for test image
img_path = 'scripts/test_images/plastic_test.jpg'

#preparing the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  
img_array /= 255.0  # Normalization (rescale)

#making prediction
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]


print(f"Model's prediction: {predicted_class}")