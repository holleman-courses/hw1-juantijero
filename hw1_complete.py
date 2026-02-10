#!/usr/bin/env python

# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras import Input, layers, Sequential

# Helper libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# --- Task 4: Simple Dense Model ---
def build_model1():
    model = Sequential(name='model1')
    # Flatten the 32x32x3 input images
    model.add(layers.Flatten(input_shape=(32, 32, 3)))
    # Three hidden layers with 128 units and LeakyReLU
    for _ in range(3):
        model.add(layers.Dense(128))
        model.add(layers.LeakyReLU())
    # Final output layer with 10 units
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# --- Task 7: Depthwise Separable Convolution Model ---
def build_model2():
    model = Sequential(name='model2')
    # Three blocks of SeparableConv2D, LeakyReLU, and MaxPooling
    # Block 1
    model.add(layers.SeparableConv2D(128, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 2
    model.add(layers.SeparableConv2D(128, (3, 3), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    # Block 3
    model.add(layers.SeparableConv2D(128, (3, 3), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# --- Task 8: Functional API with Residual Connection ---
def build_model3():
    inputs = Input(shape=(32, 32, 3))
    x = layers.Flatten()(inputs)
    # First Dense Layer
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    # Save for shortcut (Skip Connection)
    shortcut = x 
    # Second Dense Layer
    x = layers.Dense(128)(x)
    x = layers.LeakyReLU()(x)
    # Add the shortcut back
    x = layers.Add()([x, shortcut])
    
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name='model3')
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# --- Task 9: Sub-50k Parameter Model Challenge ---
def build_model50k():
    model = Sequential(name='model50k')
    # Initial extraction
    model.add(layers.SeparableConv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    # Deep features
    model.add(layers.SeparableConv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    # Final spatial processing
    model.add(layers.SeparableConv2D(64, (3, 3), padding='same'))
    model.add(layers.LeakyReLU())
    # GlobalAveragePooling reduces parameters significantly
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# --- Main Execution Block ---
if __name__ == '__main__':

    # Task 1: Load and display personal image
    try:
        my_photo = image.imread('test_image_horse.png') 
        plt.imshow(my_photo)
        plt.title("Task 1: Image Test")
        plt.axis('off') 
        #plt.show()
    except FileNotFoundError:
        print("Error: Task 1 image 'test_image_horse.png' not found.")

    # Task 5: Load and preprocess CIFAR10
    from keras.datasets import cifar10
    from sklearn.model_selection import train_test_split

    (train_images_full, train_labels_full), (test_images, test_labels) = cifar10.load_data()
    # Normalize pixel values
    train_images_full, test_images = train_images_full / 255.0, test_images / 255.0
    # Create 90/10 split
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images_full, train_labels_full, test_size=0.1, random_state=42)

    # --- Training Sequence ---
    # Model 1
    model1 = build_model1()
    model1.summary() # Expected Params: 424,074
    print("Training Model 1...")
    history1 = model1.fit(train_images, train_labels, epochs=20, 
                          validation_data=(val_images, val_labels))

    # Model 2
    model2 = build_model2()
    model2.summary() 
    print("Training Model 2...")
    history2 = model2.fit(train_images, train_labels, epochs=20, 
                          validation_data=(val_images, val_labels))

    # Model 3
    model3 = build_model3()
    model3.summary() 
    print("Training Model 3 (Residual)...")
    history3 = model3.fit(train_images, train_labels, epochs=20, 
                          validation_data=(val_images, val_labels))

    # Model 50k
    model50k = build_model50k()
    model50k.summary() # Verify Total Params < 50,000
    print("Training Model 50k...")
    history50k = model50k.fit(train_images, train_labels, epochs=20, 
                              validation_data=(val_images, val_labels))