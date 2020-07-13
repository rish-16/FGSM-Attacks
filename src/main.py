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

img = load_img('../assets/dog1.jpg', color_mode="rgb")
img = img_to_array(img)
img = preprocess_img(img)

preds = model.predict(img)
_, image_class, class_confidence = get_label(preds)
print (image_class, class_confidence)

def fgsm(x, y_adv, epsilon):
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    with tf.GradientTape() as gt:
        gt.watch(x)
        
        label = model(x)
        loss = loss_func(y_adv, label)
        print (loss)
        
    grad = gt.gradient(loss, x)
    gamma = epsilon * tf.sign(grad)
    
    return gamma
    
y_adv_label = 10
y_adv = tf.one_hot(y_adv_label, preds.shape[-1])
y_adv = tf.reshape(y_adv, shape=[1, preds.shape[-1]])

noise = fgsm(img, y_adv, 0.1)
plt.imshow(noise[0] * 0.5 + 0.5)
plt.show()

x_adv = img + noise
x_adv = tf.clip_by_value(x_adv, -1, 1)

plt.imshow(x_adv[0] * 0.5 + 0.5)
plt.show()
