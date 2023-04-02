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
        image = image.resize((256,256))
        train_images.append(np.array(image))
        train_labels.append(i)

val_images = []
val_labels = []
for i in range(len(val_path)):
    path = val_path[i]
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        image = Image.open(filepath)
        image = image.resize((256,256))
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

#divido i dati di addestramento in un due sottoinsiemi, uno di train e uno di test
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

#definisco la rete CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.LeakyReLU(alpha=0.1),
    layers.Dropout(0.5), 
    layers.Dense(len(labels), activation='softmax')
])

#definisco la fase di compilazione del modello
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Addestro il modello utilizzando i dati di addestramento
model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Valuto il modello sui dati di test
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

# Faccio predizione su un'immagine
image_path = "C:\\Lavori\\repos-AnimalML\\AnimalML\\ImagePrediction\\flickr_dog_000111.jpg"
image = Image.open(image_path)
image = np.array(image.resize((256, 256)))
image = image.astype('float32') / 255
prediction = model.predict(np.array([image]))
print("Prediction: ", labels[np.argmax(prediction)])



