import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

#Importieren des Datensatzes
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Daten aufteilen: 0.8 training 0.1 testing 0.1 validation
(rawTrain, rawValidation, rawTest), metadata = tfds-load(
    "cats_vs_dogs",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    with_info=True,
    as_supervised=True
)
getLableNAme = metadata.features["label"].int2str