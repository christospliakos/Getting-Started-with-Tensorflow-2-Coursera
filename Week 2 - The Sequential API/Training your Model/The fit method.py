import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K


model = Sequential([
    Dense(64, activation='elu', input_shape=(32, )),
    Dense(100, activation='softmax')
])

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# X_train: (num_samples, num_features)
# Y_train: (num_samples, num_classes)

history = model.fit(X_train, Y_train, epochs=10, batch_size=16)