import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Datensatz laden
numberMnist = keras.datasets.mnist

(trainImages, trainLabels), (testImages, testLabels) = numberMnist.load_data()

classNames = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

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
model.fit(trainImages, trainLabels, epochs=10)

model.save("numernErkennen.h5")

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
    plt.imshow(testImages[num], cmap="Greys", interpolation="none")   # "
    plt.grid(False)               # "
    plt.show()                    # "