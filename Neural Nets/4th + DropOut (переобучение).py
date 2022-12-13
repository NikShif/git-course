import numpy as np 
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255 
x_test = x_test/255 
y_train_cat = keras.utils.to_categorical(y_train, 10) 
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    Flatten(input_shape = (28,28,1)),
    Dense(300, activation='relu'),
    Dropout(0.8),
    Dense(10, activation='softmax')])

print(model.summary())

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

his = model.fit(x_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.2)
              
plt.plot(his.history['loss'])
plt.plot(his.history['val_loss'])
plt.show()              