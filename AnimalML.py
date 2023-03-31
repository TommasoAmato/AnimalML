import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

train_path = [
    "C:/Users/Tommaso/Documents/archive/afhq/train/cat",
    "C:/Users/Tommaso/Documents/archive/afhq/train/dog",
    "C:/Users/Tommaso/Documents/archive/afhq/train/wild"
]
val_path = [
    "C:/Users/Tommaso/Documents/archive/afhq/val/cat",
    "C:/Users/Tommaso/Documents/archive/afhq/val/dog",
    "C:/Users/Tommaso/Documents/archive/afhq/val/wild"
]

labels = ["cat", "dog", "wild"]

train_images = []
train_labels = []
for i in range(len(train_path)):
    path = train_path[i]
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        image = Image.open(filepath)
        train_images.append(np.array(image))
        train_labels.append(i)

val_images = []
val_labels = []
for i in range(len(val_path)):
    path = val_path[i]
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        image = Image.open(filepath)
        val_images.append(np.array(image))
        val_labels.append(i)
        
train_images = np.array(train_images)
train_images = train_images.astype('float32') / 255
val_images = np.array(val_images)
val_images = val_images.astype('float32') / 255

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

train_labels = keras.utils.to_categorical(train_labels, num_classes=len(labels))
val_labels = keras.utils.to_categorical(val_labels, num_classes=len(labels))

train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)


# per 8gb troppo oneroso

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(labels), activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Valuta il modello sui dati di test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)



