import sys
import numpy as np
import matplotlib.pyplot as plt

from util import extract_data, plotLoss, plotAccuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MeanSquaredError, AUC, Accuracy

if len(sys.argv) != 2:
    print("Usage: python autoencoder.py <path to train file>")
    sys.exit(-1)

inputFile = str(sys.argv[1])

data, x, y = extract_data(inputFile)

inChannel = 1
data = data.reshape(-1, x, y, inChannel)
data = data / np.max(data)


def getParameters():
    try:
        batch_size = int(input("please enter batch_size : "))
    except ValueError:
        print("batch_size must be an integer")
        sys.exit(1)

    try:
        epochs = int(input("please enter epochs number : "))
    except ValueError:
        print("epochs must be an integer")
        sys.exit(1)

    return batch_size, epochs


def plotPrediction(model, data):
    predictions = model.predict(data)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(data[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(predictions[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def getAutoencoder(
    x,
    y,
    activationFunction="softmax",
    lastActivationFunction="sigmoid",
    lossFunction="mean_squared_error",
    filters=(3, 3),
):

    input_img = Input(shape=(x, y, inChannel))

    def encoder(input_img, filters):
        conv1 = Conv2D(32, filters, activation=activationFunction, padding="same")(
            input_img
        )  # 28 x 28 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32

        conv2 = Conv2D(64, filters, activation=activationFunction, padding="same")(
            pool1
        )  # 14 x 14 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64

        conv3 = Conv2D(128, filters, activation=activationFunction, padding="same")(
            pool2
        )  # 7 x 7 x 128

        return conv3

    def decoder(input_conv, filters):
        conv3 = Conv2D(128, filters, activation=activationFunction, padding="same")(
            input_conv
        )  # 7 x 7 x 128
        up3 = UpSampling2D((2, 2))(conv3)  # 14 x 14 x 128

        conv2 = Conv2D(64, filters, activation=activationFunction, padding="same")(
            up3
        )  # 14 x 14 x 64
        up2 = UpSampling2D((2, 2))(conv2)  # 28 x 28 x 64

        covn1 = Conv2D(1, filters, activation=lastActivationFunction, padding="same")(
            up2
        )  # 28 x 28 x 1

        return covn1

    def autoencoder(input_img, filters):
        return decoder(encoder(input_img, filters), filters)

    autoencoder = Model(input_img, autoencoder(input_img, filters))
    autoencoder.compile(
        loss=lossFunction,
        optimizer=RMSprop(),
        metrics="accuracy",
    )

    return autoencoder


def newModel():
    batch_size, epochs = getParameters()
    activationFunction = "linear"
    lastActivationFunction = "linear"
    filters = (3, 3)

    autoencoder = getAutoencoder(
        x=x,
        y=y,
        activationFunction=activationFunction,
        lastActivationFunction=lastActivationFunction,
        filters=filters,
    )
    autoencoder.summary()

    train_X, valid_X, train_Y, valid_Y = train_test_split(
        data, data, test_size=0.25, shuffle=42
    )

    autoencoder_train = autoencoder.fit(
        train_X,
        train_Y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(valid_X, valid_Y),
    )

    while True:
        try:
            answer = int(
                input(
                    "Press 1 to save the model\nPress 2 to print training plots\nPress 3 to continue\n"
                )
            )
        except ValueError:
            print("Answer must be an integer")

        if answer == 1:
            autoencoder.save(str(input("Provide the name of the file: ")))
        elif answer == 2:
            plotLoss(autoencoder_train, "model_loss.png")
            plotAccuracy(autoencoder_train, "model_accuracy")
            plotPrediction(data)
        elif answer == 3:
            break
        else:
            print("Invalid input")

    return autoencoder


def getModel():
    while True:
        file = str(input("Please provide the file name: "))
        try:
            model = load_model(file)
            break
        except IOError:
            print("File name:", file, "dones't exist")
        except ImportError:
            print("File is corrupted")

    return model


if __name__ == "__main__":

    while True:
        try:
            answer = int(
                input(
                    "Press 1 to train a new model\nPress 2 to load an existing model\nPress 3 to exit\n"
                )
            )
        except ValueError:
            print("Answer must be an integer")

        if answer == 1:  # New Model
            autoencoder = newModel()
        elif answer == 2:  # Load Model
            autoencoder = getModel()
            plotPrediction(autoencoder, data)
        elif answer == 3:
            break
        else:
            print("Invalid input")
