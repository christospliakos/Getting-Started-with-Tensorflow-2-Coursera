import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


model1 = Sequential([
    Dense(64, activation='relu', input_shape=(784, )),
    Dense(10, activation='softmax')
])

# OR

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(784, )))
model.add(Dense(10, activation='softmax'))

# If we know input shape
model2 = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation='relu', input_shape=(784, )),
    Dense(10, activation='softmax')
])
