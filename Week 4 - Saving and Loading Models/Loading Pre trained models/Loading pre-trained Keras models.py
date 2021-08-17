from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet',  # Learned weights on imagenet
                 include_top=True)

img_input = image.load_img('my_picture.jpg', target_size=(224, 224))
img_input = image.img_to_array(img_input)
img_input = preprocess_input(img_input[np.newaxis, ...])

preds = model.predict(img_input)
decoded_predictions = decode_predictions(preds, top=10)[0]

print(decoded_predictions)