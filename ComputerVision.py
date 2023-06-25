import tensorflow as tf
import scipy
import numpy as np
from keras import datasets, layers, models
from keras.utils import img_to_array
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Laden und Aufteilen der Datensaetze
(trainImages, trainLabels),(testImages, testLabels) = datasets.cifar10.load_data()

# Pixelwerte zwischen 1 und 0 einschraenken
trainImages, testImages = trainImages / 255.0, testImages / 255.0

# Klassennamen
classNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# TEST
ImgIndex = 1
plt.imshow(trainImages[ImgIndex], cmap=plt.cm.binary)
plt.xlabel(classNames[trainLabels[ImgIndex][0]])
plt.show()

# Transformiert das Bild um es mehrfach nutzen zu koennen ohne Dopplungen in kauf nehmen zu muessen

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

testImg = trainImages[14]
img = img_to_array(testImg)
img = img.reshape((1,) + img.shape)

i = 0
for batch in datagen.flow(img, save_prefix="test", save_format="jpeg"):
    plt.figure(i)
    plot = plt.imshow(img_to_array(batch[0]))
    i += 1
    if i > 4:
        break

plt.show()

# Aufbau der konvolutionellen Basis des Models (Muster Finden)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))

# Hinzufuegen der dichten Schichten (Klassifizierung)

model.add(layers.Flatten())                     # Konvertiert die Pixelwertmatix in ein eindimensonales numpy Array
model.add(layers.Dense(64, activation="relu"))  # Siehe neuralNetwork l.19
model.add(layers.Dense(10))                     # Siehe neuralNetwork l.20

model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
history = model.fit(
    trainImages,
    trainLabels,
    epochs=10,
    validation_data=(testImages, testLabels)
)

testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=2)
print(testAcc)

predictions = model.predict(testImages)
using = True
print("input stop to stop testing")
while using:
  num = input("input the ID of the image you want to test: ")
  if(num == "stop"):
    using = False
  else:
    num = int(num)
    print(classNames[np.argmax(predictions[num])])
    plt.imshow(testImages[num], cmap=plt.cm.binary)
    plt.xlabel(classNames[testLabels[num][0]])
    plt.show()