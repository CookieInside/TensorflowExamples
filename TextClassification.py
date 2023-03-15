from keras.datasets import imdb
from keras.preprocessing import sequence
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAX_LENGTH = 250 # Anzahl der Wörter pro Text, weitere Wörter werden entfernt und bei einem zu korzen Text werden Wörter mit dem Wert 0 eingefügt
BATCH_SIZE = 64

(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words= VOCAB_SIZE) # Laden des Datensatzes

trainData = pad_sequences(trainData, MAX_LENGTH)
testData = pad_sequences(testData, MAX_LENGTH)

# Erstellen des Models
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"]) # Trainigsfunktionen auswählen
history = model.fit(trainData, trainLabels, epochs=10, validation_split=0.2) # Trainiren des Models
