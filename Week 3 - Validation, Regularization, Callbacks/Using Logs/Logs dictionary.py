import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

diabetes_dataset = load_diabetes()
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
# Split the data set into training and test sets

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1)

# model = Sequential([
#     Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
#     Dense(64, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     Dense(64, activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(1)
# ])
#
# # Compile
# model.compile(loss='mse',
#               optimizer='adam',
#               metrics=['mae'])


# Custom callback
class LossAndMetricCallback(tf.keras.callbacks.Callback):

    # Print the loss after every second batch in the training set
    def on_train_batch_end(self, batch, logs=None):
        if batch % 2 == 0:
            print(f"\nAfter batch {batch} the loss is {logs['loss']}")

    # Print the loss after each batch in the test set
    def on_test_batch_end(self, batch, logs=None):
        print(f"\nAfter batch {batch}, the loss is {logs['loss']}")

    # Print the loss and mean absolute error after each epoch
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch}: Average loss is {logs['loss']}, mean absolute error is {logs['mae']}")

    # Notify the user when prediction has finished on each batch
    def on_predict_batch_end(self, batch, logs=None):
        print(f"Finished prediction on batch {batch}!")


# history = model.fit(train_data, train_targets, epochs=20,
#                     batch_size=100, callbacks=[LossAndMetricCallback()], verbose=0)
#
# model_eval = model.evaluate(test_data, test_targets, batch_size=10,
#                             callbacks=[LossAndMetricCallback()], verbose=0)
#
# model_pred = model.predict(test_data, batch_size=10,
#                            callbacks=[LossAndMetricCallback()], verbose=0)

# Learning Ratte Scheduler
# Define the learning rate schedule. The tuples below are (start_epoch, new_learning_rate)

lr_schedule = [
    (4, 0.03), (7, 0.02), (11, 0.005), (15, 0.007)
]


def get_new_epoch_lr(epoch, lr):
    # Checks to see if the input epoch is listed in the learning rate schedule
    # and if so, returns index in lr_schedule
    epoch_in_sched = [i for i in range(len(lr_schedule)) if lr_schedule[i][0] == int(epoch)]
    if len(epoch_in_sched) > 0:
        # If it is, return the learning rate corresponding to the epoch
        return lr_schedule[epoch_in_sched[0]][1]
    else:
        # Otherwise, return the existing learning rate
        return lr


class LRScheduler(tf.keras.callbacks.Callback):

    def __init__(self, new_lr):
        super(LRScheduler, self).__init__()

        self.new_lr = new_lr

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Error: Optimizer does not have a learning rate')

        curr_rate = float(self.model.optimizer.lr)

        sched_rate = self.new_lr(epoch, curr_rate)

        self.model.optimizer.lr = sched_rate
        print(f"Learning rate for epoch {epoch}, is {sched_rate}")


# Build the same model as before

new_model = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model

new_model.compile(loss='mse',
                  optimizer="adam",
                  metrics=['mae', 'mse'])

# Fit the model with our learning rate scheduler callback

new_history = new_model.fit(train_data, train_targets, epochs=20,
                            batch_size=100, callbacks=[LRScheduler(get_new_epoch_lr)], verbose=False)
