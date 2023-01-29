import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Laden und Aufteilen der Datensätze
(trainImages, trainLabels),(testImages, testLabels) = datasets.cifar10.load_data()

# Pixelwerte zwischen 1 und 0 einschränken
trainImages, testImages = trainImages / 255.0, testImages / 255.0

# Klassennamen
classNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# TEST
#ImgIndex = 1
#plt.imshow(trainImages[ImgIndex], cmap=plt.cm.binary)
#plt.xlabel(classNames[trainLabels[ImgIndex][0]])
#plt.show()

# Transformiert das Bild um es mehrfach nutzen zu können ohne Dopplungen in kauf nehmen zu müssen

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Aufbau der konvolutionellen Basis des Models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# Hinzufügen der dichten Schichten

model.add(layers.Flatten())                     # Konvertiert die Pixelwertmatix in ein eindimensonales numpy Array
model.add(layers.Dense(64, activation="relu"))  # Siehe neuralNetwork l.19
model.add(layers.Dense(10))                     # Siehe neuralNetwork l.20