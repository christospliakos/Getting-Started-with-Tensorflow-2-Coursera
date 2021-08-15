import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(12, ))
])

model.compile(
    optimizer='sgd',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train)

loss, accuracy, mae = model.evaluate(X_test, y_test)

pred = model.predict(X_sample)      # (num_samples, 12)