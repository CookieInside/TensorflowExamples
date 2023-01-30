from tensorflow._api.v2.compat.v2 import size
#from prompt_toolkit.shortcuts import yes_no_dialog
#from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
#Trainings und Prüfungsdatensatz laden
dfTrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dfTest = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")
yTrain = dfTrain.pop("survived")
yTest = dfTest.pop("survived")

#Sortierund der Spalten des Datensatzes in Numerische und Kategorische Informationen
CATEGORICAL_COLUMN = ["sex", "class", "deck", "embark_town",  "alone"]
NUMERIC_COLUMN = ["age", "n_siblings_spouses", "parch", "fare"]

#"featureComlumns" mit den Spalten befüllen
featureColumns = []
for featureName in CATEGORICAL_COLUMN:
  vocabulary = dfTrain[featureName].unique()
  featureColumns.append(tf.feature_column.categorical_column_with_vocabulary_list(featureName, vocabulary))
for feature_name in NUMERIC_COLUMN:
  featureColumns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#Erstellen der Input-Funktion
def makeInputFn(dataDf, lableDf, epochNum=10, shuffle=True, batchSize=32):
  def inputFn():
    ds = tf.data.Dataset.from_tensor_slices((dict(dataDf), lableDf))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batchSize).repeat(epochNum)
    return ds
  return inputFn
print(yTest)
features = ["sex", "age", "n_siblings_spouses", "parch", "fare", "class", "deck", "embark_town",  "alone"]
predict = {}
userRes = {}
userRes["survived"] = [0]

#Eingabe speichers
print("Please type numeric values as prompted.")
for feature in features:
  val = input(feature + ": ")
  if feature in CATEGORICAL_COLUMN:
    predict[feature] = [str(val)]
  if feature in NUMERIC_COLUMN:
    predict[feature] = [float(val)]


trainInputFn = makeInputFn(dfTrain, yTrain)
testInputFn = makeInputFn(dfTest, yTest, epochNum=1, shuffle=False)
userInputFn = makeInputFn(predict, userRes, epochNum=1, shuffle=False)

linearEst = tf.estimator.LinearClassifier(feature_columns=featureColumns)

#Trainieren und Testen des Models
linearEst.train(trainInputFn)
result = linearEst.evaluate(testInputFn)


clear_output()
print("accuracy: ", result["accuracy"])

#Ergebnis der Nutzer-Eingabe ausgeben
prediction = list(linearEst.predict(userInputFn))
print("probability of survival", float(prediction[0]["probabilities"][1])*100, "%")

#Einzelne Ergebnisse ausgeben
#result = list(linearEst.predict(testInputFn))
#for i in range(len(result)):
#  print(dfTest.loc[i])
#  print(yTest.loc[i])
#  print(result[i]["probabilities"][1])
