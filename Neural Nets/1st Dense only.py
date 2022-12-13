from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense

x = np.arange(0, 101, 1)
target = 10*x + 4

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1)) 

history = model.fit(x, target, epochs=1000, verbose=1)

plt.plot(history.history['loss'])
plt.grid(True)
print(model.predict([103]))
print(model.get_weights())

a = [.9, .04, .03, .03]
loss = np.log(a)
print(loss)