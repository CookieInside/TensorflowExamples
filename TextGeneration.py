from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

pathToFile = keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt") # Eine Textdatei laden, in diesem Fall Shakespeares Coriolanus

text = open(pathToFile, "rb").read().decode(encoding="utf-8") # Text als String im read bytes (rb) Modus auslesen und als UTF-8 decodieren
print(f"Length of text: {len(text)} characters") # Länge des Textes ausgeben

print(text[:2500])

vocab = sorted(set(text)) # Erstellt eine Liste aus allem Buchstaben des Textes
char2idx = {u:i for i, u in enumerate(vocab)} # Erstellt ein Map mit Buchstaben und Index in der Liste (benötigt um vom Buchstaben zum Index zu kommen)
idx2char = np.array(vocab) # Erstellt ein Array mit allem Buchstaben des Textes (benötigt um von Index zum Buchstaben zu kommen)

def textToInt(text):
    return np.array([char2idx[c] for c in text]) # Gibt ein Array mit den Indexen der Buchstaben eines Textes zurück

def intToText(ints):
    try:
        ints = ints.numpy() # Wandle in eine numpy-Array um wenn es nicht schon eins ist
    except:
        pass
    return "".join(idx2char[ints]) # Buchstaben einzeln anhägen und dann als String zurückgeben

textAsInt = textToInt(text)

seqLength = 100 # Länge einer Trainings-Sequenz
examplesPerEpoch = len(text)//(seqLength+1) # Beispiele pro Epoche: die Länge des Textes geteilt durch die Länge einer Trainings-Sequenz plus eins

charDataset = tf.data.Dataset.from_tensor_slices(textAsInt) # Wandelt String Dataset zu Tensorflow Dataset um

sequences = charDataset.dataset.batch(seqLength+1, drop_remainder=True)