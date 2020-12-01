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
    print(
        "error command must look like this : python autoencoder.py path to train file"
    )
    sys.exit(-1)

print(str(sys.argv[1]))

inputFile = str(sys.argv[1])

data, x, y = extract_data(inputFile)

data = data.reshape(-1, x, y, 1)  # img_num * x * y * 1
data = data / np.max(data)

train_X, valid_X, train_ground, valid_ground = train_test_split(
    data, data, test_size=0.25, shuffle=42
)


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

    try:
        neurons_fc_layer = int(input("please enter number of neurons in fc layer : "))
    except ValueError:
        print("neurons_fc_layer must be an integer")
        sys.exit(1)

    return batch_size, epochs


def getAutoencoder(
    x,
    y,
    activationFunction="softmax",
    lastActivationFunction="sigmoid",
    lossFunction="mean_squared_error",
):

    inChannel = 1
    input_img = Input(shape=(x, y, inChannel))

    def encoder(input_img):
        conv1 = Conv2D(32, (3, 3), activation=activationFunction, padding="same")(
            input_img
        )  # 28 x 28 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32

        conv2 = Conv2D(64, (3, 3), activation=activationFunction, padding="same")(
            pool1
        )  # 14 x 14 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64

        conv3 = Conv2D(128, (3, 3), activation=activationFunction, padding="same")(
            pool2
        )  # 7 x 7 x 128

        return conv3

    def decoder(input_conv):
        conv3 = Conv2D(128, (3, 3), activation=activationFunction, padding="same")(
            input_conv
        )  # 7 x 7 x 128
        up3 = UpSampling2D((2, 2))(conv3)  # 14 x 14 x 128

        conv2 = Conv2D(64, (3, 3), activation=activationFunction, padding="same")(
            up3
        )  # 14 x 14 x 64
        up2 = UpSampling2D((2, 2))(conv2)  # 28 x 28 x 64

        covn1 = Conv2D(1, (3, 3), activation=lastActivationFunction, padding="same")(
            up2
        )  # 28 x 28 x 1

        return covn1

    def autoencoder(input_img):
        return decoder(encoder(input_img))

    autoencoder = Model(input_img, autoencoder(input_img))
    autoencoder.compile(
        loss=lossFunction,
        optimizer=RMSprop(),
        metrics=[MeanSquaredError(), AUC(), Accuracy()],
    )

    return autoencoder


if __name__ == "__main__":

    batch_size, epochs = getParameters()

    autoencoder = getAutoencoder(
        x=x,
        y=y,
        activationFunction="softmax",
        lastActivationFunction="sigmoid",
        lossFunction="mean_squared_error",
    )
    autoencoder.summary()

    autoencoder_train = autoencoder.fit(
        train_X, train_ground, batch_size=batch_size, epochs=epochs
    )
    plotLoss(autoencoder_train, "autoencoder_softmax_sigmoid_128_50_loss.png")
    plotAccuracy(autoencoder_train, "autoencoder_softmax_sigmoid_128_50_accuracy.png")

    results = autoencoder.evaluate(valid_X, valid_ground, batch_size=128)
    autoencoder.save("models/autoencoder_softmax_sigmoid")
    print("test loss, test acc:", results)

    pred = autoencoder.predict(data)

    while True:
        try:
            answer = int(
                input(
                    " press 1 if you want to repeat expiriment with different paremeters \n press 2 if you want to show plots \n press 3 if you want to save model with last parameters "
                )
            )
        except ValueError:
            print("answer must be an integer")
            sys.exit(1)

        if answer == 1:
            batch_size, epochs = getParameters()

            autoencoder = getAutoencoder(
                x=x,
                y=y,
                activationFunction="softmax",
                lastActivationFunction="sigmoid",
                lossFunction="mean_squared_error",
            )
            autoencoder.summary()

            autoencoder_train = autoencoder.fit(
                train_X, train_ground, batch_size=batch_size, epochs=epochs
            )

            results = autoencoder.evaluate(data, data, batch_size=128)
            print("test loss, test acc:", results)

            pred = autoencoder.predict(data)

        elif answer == 2:
            plt.figure(figsize=(20, 4))
            print("Test Images")
            for i in range(10):
                plt.subplot(2, 10, i + 1)
                plt.imshow(data[i, ..., 0], cmap="gray")
            plt.show()
            plt.savefig("original.png")
            plt.figure(figsize=(20, 4))
            print("Reconstruction of Test Images")
            for i in range(10):
                plt.subplot(2, 10, i + 1)
                plt.imshow(pred[i, ..., 0], cmap="gray")
            plt.show()
            plt.savefig("result.png")

        elif answer == 3:
            # havent tested the # ones
            # try:
            #     answer = str(input("please give path : "))
            # except ValueError:
            #     print ("path must be a string")
            #     sys.exit(1)
            # autoencoder.save(path)
            break

        else:
            sys.exit(1)
