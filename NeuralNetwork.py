<<<<<<< HEAD:neuralNetwork.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Datensatz laden
fashionMnist = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = fashionMnist.load_data()

trainImages = trainImages / 255.0
testImages = testImages / 255.0

classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Erstellen des Modellen (Neuronales Netzwert)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),         # Input Layer, 28x28 Matrix wird in eine Reihe aus 784 Werten umgewandelt , die den Input darstellen
    keras.layers.Dense(128, activation="relu"),         # Hidden Layer, eine Schicht aus 128 nodes mit der Aktivierungsfunktion ReLU (Rectified Linear Unit) -> gibt den input Wert zurück, insofern dieser größer Null ist, alternativ wird Null zurückgegeben
    keras.layers.Dense(10, activation="softmax")        # Output Layer, eine Schicht aus 10 nodes, die für die verschiedenen Klassen stehen und softmax als Aktivierungsfunktion nutzen (softmax begrenzt den Wertebereich auf 0 bis 1 und stellt sicher, dass die Werte in Summe im ursprünglichen Verhältnis 1 ergeben)
])

model.compile(
    optimizer="adam", # Funktion die die Gewichtung im Netz bearbeitet
    loss="sparse_categorical_crossentropy", # Funktion die einen Score für das aktuelle Model berechnet und somit die stärke der Abänderungen bestimmt
    metrics=["accuracy"] # Output des Models
)

# Zuschneiden / Trainieren des Models auf den Datensatz
model.fit(trainImages, trainLabels, epochs=7)

# Model Evaluation (Test-Datensatz)
testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=1)

# Ergebnisse ausgeben
print("Test accuracy: ", testAcc)


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
    plt.figure()                  # Zeigen des Bildes
    plt.imshow(testImages[num])   # "
    plt.colorbar()                # "
    plt.grid(False)               # "
    plt.show()                    # "
    
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Datensatz laden
fashionMnist = keras.datasets.fashion_mnist

(trainImages, trainLabels), (testImages, testLabels) = fashionMnist.load_data()

trainImages = trainImages / 255.0
testImages = testImages / 255.0

classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Erstellen des Modellen (Neuronales Netzwert)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),         # Input Layer, 28x28 Matrix wird in eine Reihe aus 784 Werten umgewandelt , die den Input darstellen
    keras.layers.Dense(128, activation="relu"),         # Hidden Layer, eine Schicht aus 128 nodes mit der Aktivierungsfunktion ReLU (Rectified Linear Unit) -> gibt den input Wert zurück, insofern dieser größer Null ist, alternativ wird Null zurückgegeben
    keras.layers.Dense(10, activation="softmax")        # Output Layer, eine Schicht aus 10 nodes, die für die verschiedenen Klassen stehen und softmax als Aktivierungsfunktion nutzen (softmax begrenzt den Wertebereich auf 0 bis 1 und stellt sicher, dass die Werte in Summe im ursprünglichen Verhältnis 1 ergeben)
])

model.compile(
    optimizer="adam", # Funktion die die Gewichtung im Netz bearbeitet
    loss="sparse_categorical_crossentropy", # Funktion die einen Score für das aktuelle Model berechnet und somit die stärke der Abänderungen bestimmt
    metrics=["accuracy"] # Output des Models
)

# Zuschneiden / Trainieren des Models auf den Datensatz
model.fit(trainImages, trainLabels, epochs=7)

# Model Evaluation (Test-Datensatz)
testLoss, testAcc = model.evaluate(testImages, testLabels, verbose=1)

# Ergebnisse ausgeben
print("Test accuracy: ", testAcc)


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
    plt.figure()                  # Zeigen des Bildes
    plt.imshow(testImages[num])   # "
    plt.colorbar()                # "
    plt.grid(False)               # "
    plt.show()                    # "
>>>>>>> 2621f2ff0ce54ed5d70ba7bc2979561c70db34b3:NeuralNetwork.py
