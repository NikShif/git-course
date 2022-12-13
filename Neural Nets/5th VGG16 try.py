import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from io import BytesIO
from PIL import Image

model = keras.applications.VGG16()
print(model.summary())

# tf.keras.applications.vgg16.VGG16(
#     include_top=True,
#     weights='imagenet',
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation='softmax')

img = Image.open('rr.jpg')
img = img.resize((224, 224))
print(img.size)



# приводим к входному формату VGG-сети
img = np.array(img)
print(img.shape)
x = keras.applications.vgg16.preprocess_input(img)
print(x.shape)
x = np.expand_dims(x, axis=0)

# прогоняем через сеть
res = model.predict(x)
print(np.argmax(res))
