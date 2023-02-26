import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

#Importieren des Datensatzes
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Daten aufteilen: 0.8 training 0.1 testing 0.1 validation
(rawTrain, rawValidation, rawTest), metadata = tfds.load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    with_info=True,
    as_supervised=True
)
getLabelName = metadata.features["label"].int2str

# Funktion zum Strecken der Bilder auf eine festgesetzte Größe
IMG_SIZE = 160

def formatDatapoint(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255) # image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = rawTrain.map(formatDatapoint)
validation = rawValidation.map(formatDatapoint)
test = rawTest.map(formatDatapoint)

# Bild als Test zeigen
#for image, label in train.take(2):
#    plt.figure()
#    plt.imshow(image)
#    plt.title(getLabelName(label))
#    plt.show()

# Laden des Pretrainierten Modells
baseModel = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")

baseModel.trainable = False

globalAverageLayer = tf.keras.layers.GlobalAveragePooling2D()

perdictionLayer = keras.layers.Dense(1)

model = tf.keras.Sequential([
    baseModel,
    globalAverageLayer,
    perdictionLayer
])

# Stärke der Änderungen die den Model zugefügt werden
baseLearningRate = 0.0001

# Trainingseinstellungen des Models
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(lr=baseLearningRate),
    loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Evaluiren des Models
initialEpochs = 3
validationSteps = 20
loss0, accuracy0 = model.evaluate(validation, steps=validationSteps)

history = model.fit(
    train_batches,
    epochs=initialEpochs,
    validation_data=validation_batches
)

acc = history.history["accuracy"]
print(acc)

# Speichern des Models
model.save("hundeUndKatzen.h5")

# Laden des Models
newModel = tf.keras.models.load_model("hundeUndKatzen.h5")