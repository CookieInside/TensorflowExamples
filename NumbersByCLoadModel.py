import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#Datensatz laden
numberMnist = keras.datasets.mnist

(trainImages, trainLabels), (testImages, testLabels) = numberMnist.load_data()

classNames = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

model = tf.keras.models.load_model("./models/nummernErkennen.h5")

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