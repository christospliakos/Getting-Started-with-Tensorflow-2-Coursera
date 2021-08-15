import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import Callback
import numpy as np


class MyCallback(Callback):

    def on_train_begin(self, logs=None):
        # Do something at the start of training
        pass

    def on_train_batch_begin(self, batch, logs=None):
        # Do sth at the start of every batch iteration
        pass

    def on_epoch_end(self, epoch, logs=None):
        # Do sth at the end of every epoch
        pass


model.fit(X, Y, epochs=5, callbacks=[MyCallback()])