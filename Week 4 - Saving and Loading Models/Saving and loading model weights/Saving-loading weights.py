from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
    Dense(64, activation='sigmoid', input_shape=(10, )),
    Dense(1)
])

model.compile(optimizer='sgd', loss=BinaryCrossentropy(from_logits=True))

checkpoint = ModelCheckpoint('my_model', save_weights_only=True)

model.fit(X, y, epochs=10, callbacks=[checkpoint])

# Load weights
model.load_weights('my_model')
