import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D
import matplotlib.pyplot as plt


model = Sequential([
    Conv1D(filters=16, kernel_size=3, input_shape=(128, 64),
           kernel_initializer="random_uniform", bias_initializer="zeros",
           activation='relu'),
    MaxPooling1D(pool_size=4),
    Flatten(),
    Dense(64, kernel_initializer='he_uniform', bias_initializer='ones',
          activation='relu')
])

# Adding layers

model.add(Dense(64,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.4),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),
                activation='relu'))

model.add(Dense(8,
                kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0, seed=None),
                bias_initializer=tf.keras.initializers.Constant(value=0.4),
                activation='relu'))

# Custom weights and biases
import tensorflow.keras.backend as K


def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)


model.add(Dense(64, kernel_initializer=my_init))

print(model.summary())

# Visualizing the initialised weights and biases
fig, axes = plt.subplots(5, 2, figsize=(12, 16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

weights_layers = [layer for layer in model.layers if len(layer.weights) > 0]

for i, layer in enumerate(weights_layers):
    for j in [0, 1]:
        axes[i, j].hist(layer.weights[j].numpy().flatten(), align='left')
        axes[i, j].set_title(layer.weights[j].name)

plt.show()