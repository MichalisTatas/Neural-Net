import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from util import extract_data, plotModelLoss
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.metrics import MeanSquaredError, AUC, Accuracy

inputFile = ""

try:
    opts, args = getopt.getopt(sys.argv, "p:", ["help"])
except getopt.GetoptError:
    print("usage: autoencoder.py -p <dataset>")
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-p"):
        inputfile = arg
    elif opt in ("--help"):
        print("usage: autoencoder.py -p <dataset>")
        sys.exit(1)


data, x, y = extract_data(inputFile)

data = data.reshape(-1, x, y, 1)  # img_num * x * y * 1
data = data / np.max(data)

train_X, valid_X, train_ground, valid_ground = train_test_split(
    data, data, test_size=0.25, shuffle=42
)


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


batch_size = 128
epochs = 50
# autoencoder = getAutoencoder(
#     x=x, y=y, activationFunction="softmax", lastActivationFunction="sigmoid", lossFunction="mean_squared_error")
# autoencoder.summary()

# autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,
#                                     epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

# autoencoder.save("models/autoencoder_softmax_sigmoid")
# ploting loss graph
# plotModelLoss(autoencoder_train, epochs,
#               "models/model_softmax_sigmoid/plot.png")

autoencoder = load_model("models/autoencoder_softmax_sigmoid")

results = autoencoder.evaluate(data, data, batch_size=128)
print("test loss, test acc:", results)

pred = autoencoder.predict(data)
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
