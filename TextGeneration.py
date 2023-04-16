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

sequences = charDataset.batch(seqLength+1, drop_remainder=True) # Die letzten Zeichen des Datensatzes, die keine ganze Sequenz auffüllen können, werden nicht genutzt

def splitToInputAndTarget(chunk): # Beispiel: chunk = "hello"
    input = chunk[:-1]            #           input = "hell"
    target = chunk[1:]            #          target = "ello"
    return input, target

dataset = sequences.map(splitToInputAndTarget) # Split-Funktion auf den Datensatz anwenden
# Der Datensatz hat nun sowohl einen input als auch einen Target Anteil: der input Anteil ist der gegebene Wert und Target der Zielwert

# Erstellen von Training-Batches
BATCH_SIZE = 64 # Größe eines Batches
VOCAB_SIZE = len(vocab) # anzahl aller Charakter
EMBEDDING_DIM = 256 # Anzahl der Dimensionen des Vektors der die Worte repräsentiert
RNN_UNITS = 1024 # 
BUFFER_SIZE = 10000 # Größe des Buffers: Tensorflow speichert das Dataset in einem Buffer, da es von einem Theoretisch unendlich großem Dataset ausgeht und somit die Sequenz nicht im Ram shuffelt

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True) # Datensatz shuffeln (in zufällige Reihenfolge bringen)
# Nach dieser Interaktion mit dem Datensatz gibt es Batches, die aus 64 Sequenzen bestehen

# Modell bauen:
# Das Modell-Bauen wird mit einer Methode ungesetzt, damit zum Trainieren ein Modell erstellt werden kann, dass direkt 64 Sequenzen auf einmal verarbeitet, später, bei der Nutzung des Modells wird auf das Trainierte Modell zurückgegriffen, dieses jedoch neu konstruiert um immer nur eine Sequenz auf einmal zu verarbeiten
def buildModel(vocabSize, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocabSize,
            embeddingDim,
            batch_input_shape=[batchSize, None] # Die Batch-Größe ist bekannt, die Sequenz-Größe jedoch nicht
        ),
        tf.keras.layers.LSTM(
            rnnUnits,
            return_sequences=True, # Alle vorherrgesagten Sequenzen werden zurückgegeben
            stateful=True,
            recurrent_initializer="glorot_uniform"
        ),
        tf.keras.layers.Dense(vocabSize)    # Output-Layer: Wahrscheinlichkeit für jeden Charakter
    ])
    return model

model = buildModel(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)

for inputExampleBatch, targetExampleBatch in data.take(1):
    exampleBatchPredictions = model(inputExampleBatch) # Das Modell einen Wert vorherrsagen lassen
    #print(exampleBatchPredictions.shape, "# (batch_size, sequenze_length, vocab_size)")

#print(len(exampleBatchPredictions))
#print(exampleBatchPredictions)

#pred = exampleBatchPredictions[0]
#print(len(pred))
#print(pred)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)    # Keras loss-Funktion nutzen um eigene loss Funktion zu erstellen

model.compile(optimizer="adam", loss=loss)

checkpointDir = "./trainingCheckpoints"     # Speicherort für die Checkpoints
checkpointPrefix = os.path.join(checkpointDir, "ckpt_{epoch}")      # Name der einzelnen Checkpoints


# Speichern der Checkpoints
checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix,
    save_weights_only=True
)

# Training
history = model.fit(data, epochs=40, callbacks=[checkpointCallback])