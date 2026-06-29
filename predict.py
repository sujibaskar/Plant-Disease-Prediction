import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json

model = tf.keras.models.load_model("models/plant_model.h5")

with open("models/class_names.json","r") as f:
    classes = json.load(f)

def predict(img_path):

    img = image.load_img(img_path,target_size=(128,128))

    img = image.img_to_array(img)

    img = img/255.0

    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img,verbose=0)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    return classes[index],confidence