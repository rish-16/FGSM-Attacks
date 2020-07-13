import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

model = ResNet50(include_top=True, weights="imagenet")
model.trainable = False
    
def preprocess_img(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)
    image = image[None, ...]
    
    return image
    
def get_label(logits):
    label = decode_predictions(logits, top=1)[0][0]
    return label
    
img = load_img("../assets/dog1.jpg")
img = img_to_array(img)
img = preprocess_img(img)

preds = model.predict(img)
_, image_class, class_confidence = get_label(preds)
print (image_class, class_confidence)

def fgsm(x, y_adv, epsilon):
    with tf.GradientTape() as gt:
        gt.watch(x)
        
        label = tf.reshape(model(x), shape=[1, 10])
        loss = tf.keras.losses.categorical_crossentropy(y_adv, label)
        
    grad = gt.gradient(loss, x)
    gamma = epsilon * tf.sign(grad)
    
    return gamma
    
