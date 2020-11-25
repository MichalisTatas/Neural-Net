import tensorflow as tf
import numpy as np

from util import extract_data, extract_labels
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MeanSquaredError, AUC, Accuracy

test_data, x, y = extract_data("data/t10k-images-idx3-ubyte")
test_labels = extract_labels("data/t10k-labels-idx1-ubyte")

inChannel = 1
test_data = test_data.reshape(-1, x, y, inChannel)
test_data = test_data / np.max(test_data)
test_labels = test_labels.reshape(-1, 1)

train_X, valid_X, train_ground, valid_ground = train_test_split(
    test_data, test_labels, test_size=0.25, shuffle=42)

activationFunction = "sigmoid"
lastActivationFunction = "sigmoid"
lossFunction = "mean_squared_error"

autoencoder = load_model("models/autoencoder_softmax_sigmoid")

model = Sequential()
for layer in autoencoder.layers[:6]:
    model.add(layer)

model.add(Dense(32, activation=activationFunction))  # how many layers?
model.add(Dense(10, activation=lastActivationFunction))
model.summary()
model.compile(loss=lossFunction, optimizer=RMSprop(),
              metrics=[MeanSquaredError()])

batch_size = 128
epochs = 50
model_train = model.fit(train_X, train_ground, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))
