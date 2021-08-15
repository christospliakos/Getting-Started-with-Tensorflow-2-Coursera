import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv1D, MaxPooling1D

model = Sequential([
    Dense(64, activation='elu', input_shape=(32, )),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='sgd',             # 'adam', 'rmsprop', 'adadelta'
    loss='binary_crossentropy',  # 'mean_squared_error', 'categorical_crossentropy'
    metrics=['accuracy', 'mae']
)

# OR we can use the objects and pass some arguments:
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.7), tf.keras.metrics.MeanAbsoluteError()]
)
