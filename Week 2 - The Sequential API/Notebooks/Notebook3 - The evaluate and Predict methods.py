import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Softmax, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential


fashion_mnist_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()


model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((3, 3)),
    Flatten(),
    Dense(10, activation='softmax')
])

opt = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', 'mae']
)

print("Training set shape: ", train_images.shape)

labels = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
]

print("Label of the first train image: ", train_labels[0])

# Rescalling images
train_images = train_images / 255
test_images = test_images / 255

# Display one image
# i = 0
# img = train_images[i, :, :]
# plt.imshow(img)
# plt.title(f"Label: {labels[train_labels[i]]}")
# plt.show()

# Fit the model
# We must add a dummy channel to the images
history = model.fit(
    train_images[..., np.newaxis], train_labels, epochs=4, batch_size=256
)

# Plot training history

df = pd.DataFrame(history.history)
print(df.head())

# # Plot of the loss
# loss_plot = df.plot(y='loss', title='Loss vs Epochs', legend=False)
# loss_plot.set(xlabel='Epochs', ylabel='Loss')
#
# # Plot of accuracy
# acc_plot = df.plot(y='accuracy', title='Acc vs Epochs', legend=False)
# acc_plot.set(xlabel='Epochs', ylabel='Acc')
#
# # Plot of mae
# mae_plot = df.plot(y='mae', title='MAE vs Epochs', legend=False)
# mae_plot.set(xlabel='Epochs', ylabel='MAE')
#
# plt.show()

# Evaluate model
test_lost, test_acc, test_mae = model.evaluate(test_images[..., np.newaxis], test_labels, verbose=2)

# Prediction
inx = np.random.choice(test_images.shape[0])

test_image = test_images[inx]
plt.imshow(test_image)
plt.title(f"Label: {labels[test_labels[inx]]}")

predictions = model.predict(test_image[np.newaxis, ..., np.newaxis])
print(f"Model prediction: {labels[np.argmax(predictions)]}")
plt.show()