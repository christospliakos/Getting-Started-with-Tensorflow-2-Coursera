import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint

# Import the CIFAR-10 dataset and rescale the pixel values

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Use smaller subset -- speeds things up
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# # Plot the first 10 CIFAR-10 images
# fig, ax = plt.subplots(1, 10, figsize=(10, 1))
# for i in range(10):
#     ax[i].set_axis_off()
#     ax[i].imshow(x_train[i])


# Introduce function to test model accuracy

def get_test_accuracy(model_, x_test_, y_test_):
    test_loss, test_acc = model_.evaluate(x_test_, y_test_, verbose=0)
    print('accuracy: {acc:0.3f}'.format(acc=test_acc))


# Introduce function that creates a new instance of a simple CNN

def get_new_model():
    model_ = Sequential([
        Conv2D(filters=16, input_shape=(32, 32, 3), kernel_size=(3, 3),
               activation='relu', name='conv_1'),
        Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
        MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32, activation='relu', name='dense_1'),
        Dense(units=10, activation='softmax', name='dense_2')
    ])
    model_.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model_


model = get_new_model()
print(model.summary())

# Untrained accuracy
get_test_accuracy(model, x_test, y_test)

checkpoint_path = 'model_checkpoints/checkpoint'
checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                             frequency='epoch',
                             save_weights_only=True,
                             verbose=1)

model.fit(x_train, y_train, epochs=3,
          callbacks=[checkpoint])

# Create a new instance of the (initialised) model, accuracy around 10% again
model = get_new_model()
get_test_accuracy(model, x_test, y_test)

# Load weights -- accuracy is the same as the trained model
model.load_weights(checkpoint_path)
get_test_accuracy(model, x_test, y_test)