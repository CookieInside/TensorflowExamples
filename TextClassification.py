from keras.datasets import imdb
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584
MAX_LENGTH = 250 # Anzahl der Wörter pro Text, weitere Wörter werden entfernt und bei einem zu korzen Text werden Wörter mit dem Wert 0 eingefügt
BATCH_SIZE = 64

# Text Kodierer
wordIndex = imdb.get_word_index() # Liste aller Wörter die im Training genutzt worden

def encodeText(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text) # Umwandeln des Strings in eine Liste aus Worten
    tokens = [wordIndex[word] if word in wordIndex else 0 for word in tokens] # Wörter durch Nummern ersetzten
    return pad_sequences([tokens], MAX_LENGTH)[0]

# Text Dekodierer
reverseWordIndex = {value: key for (key, value) in wordIndex.items()} # eine Map mit Nummer und entsprechenden Wort

def decodeInt(integers):
    PAD = 0 # Null ist der Füller-Wert
    text = "" # Basis des Strings
    for num in integers: # Durch die Liste an Nummern loopen
        if num != PAD: # wenn es kein Füller-Wert ist
            text += reverseWordIndex[num] + " " # Suche das Wort für den Wert in der Map
    return text[:-1] # Gib alles zurück bis auf das letzte Zeichen (hier ein unnötiges Leerzeichen)


def trainModel():
    (trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words= VOCAB_SIZE) # Laden des Datensatzes
    trainData = pad_sequences(trainData, MAX_LENGTH)
    testData = pad_sequences(testData, MAX_LENGTH)
    print(decodeInt(testData[0]))
    # Erstellen des Models
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"]) # Trainigsfunktionen auswählen
    history = model.fit(trainData, trainLabels, epochs=10, validation_split=0.2, batch_size=BATCH_SIZE) # Trainiren des Models

    results = model.evaluate(testData, testLabels) # Model anhand des Test-Datensatzen testen
    print(results) # Test-Ergebnisse ausgeben
    model.save("./models/textClassification.h5")
#trainModel()

def predict(text):
    encoded = encodeText(text) # text kodieren
    pred = np.zeros((1,250)) # Basis Array erstellen
    pred[0] = encoded # Text in Array einfügen
    result = model.predict(pred) # Probe mit Model vorhersagen lassen
    if result[0][0] > 0.5:
        print(f"positive ({result[0][0]})")
    else:
        print(f"negative ({result[0][0]})")

model = tf.keras.models.load_model("./models/textClassification.h5")

using = True
print("input 'stop' to stop testing, the output number means positive if it is greater than .5 and negative if it is smaller")
while using:
    text = input("Input your movie review: ")
    if(text == "stop"):
        using = False
    else:
        predict(text)