from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

model = Sequential([
    Dense(128, activation='relu'),
    Dense(2)
])

opt = Adam(learning_rate=0.05)
model.compile(
    optimizer=opt,
    loss='mse',
    metrics=['mape']
)

history = model.fit(inputs, targets, validation_split=0.2)

# OR
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

history2 = model.fit(X_train, y_train, validation_data=(X_test, y_test))
