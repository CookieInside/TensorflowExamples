import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions

#Init-Distribution, der Start der "Simulation" bzw. Vorhersage
initialDistribution = tfd.Categorical(probs=[0.8, 0.2])

#Transistion-Distribution, alle Vorhersagen die auf eine andere Vorhersage folgen
transitionDistribution = tfd.Categorical(probs=[[0.5, 0.5],[0.5, 0.5]])

#Observations-Distribution, eigentschaften die den verschiedenen Stati gegeben werden (loc = Durchschnitt / scale = Maximale Abweichung vom Durchschnitt)
observationDistribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

#Erstellen des Models
model = tfd.HiddenMarkovModel(
    initial_distribution=initialDistribution,
    transition_distribution=transitionDistribution,
    observation_distribution=observationDistribution,
    num_steps=7
)

#Das Model berechnet die Wahrscheinlichkeiten
mean = model.mean()

#Konsolen-Output
with tf.compat.v1.Session() as sess:
  print(mean.numpy())