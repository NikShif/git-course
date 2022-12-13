import numpy as np
import matplotlib.pyplot as plt
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss':[],'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.history['val_loss'].append(logs.get('val_loss'))

history = LossHistory()

model = keras.Sequential()
model.add(keras.layers.Dense(1, activation='relu', input_dim=1))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy')

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, ))
print(labels)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32, 
          validation_split=0.2, callbacks=[history])

# Plot the history
y1=history.history['loss']
y2=history.history['val_loss']
x1 = np.arange(len(y1))
k=len(y1)/len(y2)
x2 = np.arange(k,len(y1)+1,k)
fig, ax = plt.subplots(figsize=(10,10), tight_layout=True)
line1, = ax.plot(x1, y1, label='loss', linewidth=2, linestyle=':')
line2, = ax.plot(x2, y2, label='val_loss')
