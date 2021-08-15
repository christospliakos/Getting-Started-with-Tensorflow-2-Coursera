import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# L2 regularization
model = Sequential([
    Dense(64, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dense(1, activation='sigmoid')
])

# OR - L1 regularization

model2 = Sequential([
    Dense(64, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l1(0.005)),
    Dense(1, activation='sigmoid')
])

# OR - Both

model3 = Sequential([
    Dense(64, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.005, l2=0.001)),
    Dense(1, activation='sigmoid')
])

# Bias regularizer

model4 = Sequential([
    Dense(64, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.005, l2=0.001),
          bias_regularizer=tf.keras.regularizers.l1(0.005)),
    Dense(1, activation='sigmoid')
])

# Dropout
model5 = Sequential([
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])