#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd

CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]

#mit keras Datensätze speichern
trainPath = tf.keras.utils.get_file("irisTraining.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
testPath = tf.keras.utils.get_file("irisTest.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(trainPath, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(testPath, names=CSV_COLUMN_NAMES, header=0)
trainY = train.pop("Species")
testY = test.pop("Species")

#Input Funktion
def inputFn(features, labels, training=True, batchSize=256):
  #Inputs zu Dataset konvertieren
  dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
  #im Trainingsmodus umsoriterien und wiederholen
  if training:
    dataset = dataset.shuffle(1000).repeat()
  return dataset.batch(batchSize)

#Erstellen einer Liste mit den Spaltennamen der Gegebenen Informationen
myFeatureColumns = []
for key in train.keys():  #geht durch alle Spaltennamen des Dataframes
  myFeatureColumns.append(tf.feature_column.numeric_column(key=key)) #Spaltennamen der Numerischen Werte hinzufügen

#Aufbau eines Deep-Neural-Networks (DNN) mit 2 hidden layers mit jeweils 30 und 10 hidden nodes
classifier = tf.estimator.DNNClassifier(
    feature_columns=myFeatureColumns, #Übergabe der Features (Spaltennamen der gegebenen Informationen)
    hidden_units=[30,10], #zwei hidden layer mit 30 und 10 hidden nodes erstellen
    n_classes=3 #Anzahl der Klaasen (hier die Spezies)
)

#trainieren des Models
classifier.train(
    input_fn=lambda: inputFn(train, trainY, training=True),  #lambda gibt den nach dem Doppelpunkt in der selben Zeile folgenden code als Funktion zurück
    steps=5000 #Anzahl der Elemente die betrachtet werden sollen
)

testResult = classifier.evaluate(input_fn=lambda: inputFn(test, testY, training=False))
print("\nTest set accuracy: {accuracy:0.3f}\n".format(**testResult))

#||||Nutzer-Eingabe von Werten||||
def inputFn(features, batchSize=256): #Angepasste Input-Funktion für Nutzer-Einagben
  return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batchSize)

features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
predict = {}

#Eingabe speichers
print("Please type numeric values as prompted.")
for feature in features:
  valid = True
  while valid:
    val = input(feature + ": ")
    if not val.isdigit():
      valid = False
  predict[feature] = [float(val)] #Eingabe als float in einem dataframe speichern
#Eingabe mit dem Model verarbeiten und eine Vorhersage treffen
predictions = classifier.predict(input_fn=lambda: inputFn(predict))
for predDict in predictions:
  classID = predDict["class_ids"][0]
  probability = predDict["probabilities"][classID]
  print('Prediction is "{}" /{:.1f})'.format(SPECIES[classID], 100*probability) + "%") #Ausgabe der Vorherrsage