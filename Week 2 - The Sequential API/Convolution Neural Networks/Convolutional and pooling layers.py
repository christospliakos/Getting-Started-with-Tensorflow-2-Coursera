import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # (None, 30, 30, 16)
    MaxPooling2D((3, 3)),                                            # (None, 10, 10, 16)
    Flatten(),                                                       # (None, 1600)
    Dense(64, activation='relu'),                                    # (None, 64)
    Dense(10, activation='softmax')                                  # (None, 10)
])

print(model.summary())

model2 = Sequential([
    Conv2D(16, kernel_size=3, padding='SAME', activation='relu',
           input_shape=(32, 32, 3)),                                 # (None, 32, 32, 16)
    MaxPooling2D((3, 3)),                                            # (None, 10, 10, 16)
    Flatten(),                                                       # (None, 1600)
    Dense(64, activation='relu'),                                    # (None, 64)
    Dense(10, activation='softmax')                                  # (None, 10)
])

print(model2.summary())
