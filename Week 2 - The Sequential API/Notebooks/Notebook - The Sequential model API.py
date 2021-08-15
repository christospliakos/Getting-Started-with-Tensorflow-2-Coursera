import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


# Building a Sequential Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(16, activation='relu', name='layer_1'),
    Dense(16, activation='relu'),
    Dense(10, activation='softmax'),
    Dense(10),
    Softmax()
])

# print(model.weights)
# print(model.summary())

# Building a Convolutional NN
model2 = Sequential([
    Conv2D(16, kernel_size=3, activation='relu', input_shape=(1, 28, 28), data_format='channels_first'),
    MaxPooling2D(pool_size=3, data_format='channels_first'),
    Flatten(),
    Dense(10, activation='softmax')
])

# print(model2.summary())

# The compile method
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
mae = tf.keras.metrics.MeanAbsoluteError()

model2.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=[acc, mae]
)

print(model2.optimizer)
print(model2.loss)
print(model2.metrics)
print(model2.optimizer.lr)
