import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']

# Normalize
targets = (targets - targets.mean(axis=0)) / targets.std()

# Split data
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

# Build model
model = Sequential([
    Dense(64, input_shape=[train_data.shape[1], ], activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu')
])

# Add a customised batch normalisation layer
model.add(tf.keras.layers.BatchNormalization(
    momentum=0.95,
    epsilon=0.005,
    axis = -1,
    beta_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    gamma_initializer=tf.keras.initializers.Constant(value=0.9)
))

# Add the output layer
model.add(Dense(1))

# Compile the model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(train_data, train_targets,
                    epochs=100,
                    validation_split=0.15,
                    batch_size=64,
                    verbose=False)


# Plot the learning curves

frame = pd.DataFrame(history.history)
epochs = np.arange(len(frame))

fig = plt.figure(figsize=(12,4))

# Loss plot
ax = fig.add_subplot(121)
ax.plot(epochs, frame['loss'], label="Train")
ax.plot(epochs, frame['val_loss'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Loss vs Epochs")
ax.legend()

# Accuracy plot
ax = fig.add_subplot(122)
ax.plot(epochs, frame['mae'], label="Train")
ax.plot(epochs, frame['val_mae'], label="Validation")
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Mean Absolute Error vs Epochs")
ax.legend()

plt.show()