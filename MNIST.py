


from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Load the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train[333]

y_train[333]

plt.imshow(x_train[333], cmap='Greys_r')

print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

x_train = x_train.reshape(len(x_train), 28*28)
x_test = x_test.reshape(len(x_test), 28*28)

# the numbers to floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

## Normalize the inputs
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices (one-hot encoding)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[333]

model_4 = Sequential()
model_4.add(Dense(64, activation='relu', input_shape=(784,)))
model_4.add(Dropout(0.3)) # dropout 0.3
model_4.add(Dense(64, activation='relu'))
model_4.add(Dropout(0.3))
model_4.add(Dense(10, activation='softmax')) # using softmax function with 10 hidden units.

model_4.summary()

# optimizer with adam (Adapt Moment estimation)
model_4.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])


batch_size = 64  # Batch size 64
epochs = 25     #epochs 25
history = model_4.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))


score = model_4.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)


plot_loss_accuracy(history)