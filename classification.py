import tensorflow as tf
import numpy as np

from util import extract_data, extract_labels
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import Accuracy

test_data, x, y = extract_data("data/t10k-images-idx3-ubyte")
test_labels = extract_labels("data/t10k-labels-idx1-ubyte")

inChannel = 1
test_data = test_data.reshape(-1, x, y, inChannel)
test_data = test_data / np.max(test_data)
test_labels = test_labels.reshape(-1, 1)

train_X, valid_X, train_ground, valid_ground = train_test_split(
    test_data, test_labels, test_size=0.25, shuffle=42)

print(train_X.shape)
print(train_ground.shape)

activationFunction = "sigmoid"
lastActivationFunction = "sigmoid"
lossFunction = "mean_squared_error"


def get_model(input_img, autoencoder=""):
    conv1 = Conv2D(32, (3, 3), activation=activationFunction,
                   padding='same')(input_img)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation=activationFunction,
                   padding='same')(pool1)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation=activationFunction,
                   padding='same')(pool2)

    flat = Flatten()(conv3)
    dense = Dense(1000, activation=activationFunction)(flat)
    output = Dense(1, activation=activationFunction)(dense)

    return output


autoencoder = load_model("models/autoencoder_softmax_sigmoid")

input_img = Input(shape=(x, y, inChannel))
model = Model(input_img, get_model(input_img, autoencoder))
model.summary()

model.layers[1].set_weights(autoencoder.layers[1].get_weights())
model.layers[3].set_weights(autoencoder.layers[3].get_weights())
model.layers[5].set_weights(autoencoder.layers[5].get_weights())

model.layers[1].trainable = False
model.layers[3].trainable = False
model.layers[5].trainable = False

model.compile(loss=lossFunction, optimizer=RMSprop(),
              metrics=Accuracy())

batch_size = 128
epochs = 5
model_train = model.fit(train_X, train_ground, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))


epochs = 5
model.layers[1].trainable = True
model.layers[3].trainable = True
model.layers[5].trainable = True

model_train = model.fit(train_X, train_ground, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))
