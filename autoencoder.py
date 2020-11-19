import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop


def getAutoencoder(x, y,
                   activationFunction="softmax", lastActivationFunction="sigmoid", lossFunction="mean_squared_error"):

    inChannel = 1
    input_img = Input(shape=(x, y, inChannel))

    def encoder(input_img):
        conv1 = Conv2D(32, (3, 3), activation=activationFunction,
                       padding='same')(input_img)  # 28 x 28 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32

        conv2 = Conv2D(64, (3, 3), activation=activationFunction,
                       padding='same')(pool1)  # 14 x 14 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64

        conv3 = Conv2D(128, (3, 3), activation=activationFunction,
                       padding='same')(pool2)  # 7 x 7 x 128

        return conv3

    def decoder(input_conv):
        conv3 = Conv2D(128, (3, 3), activation=activationFunction,
                       padding='same')(input_conv)  # 7 x 7 x 128
        up3 = UpSampling2D((2, 2))(conv3)  # 14 x 14 x 128

        conv2 = Conv2D(64, (3, 3), activation=activationFunction,
                       padding='same')(up3)  # 14 x 14 x 64
        up2 = UpSampling2D((2, 2))(conv2)  # 28 x 28 x 64

        covn1 = Conv2D(1, (3, 3), activation=lastActivationFunction,
                       padding='same')(up2)  # 28 x 28 x 1

        return covn1

    def autoencoder(input_img):
        return decoder(encoder(input_img))

    autoencoder = Model(input_img, autoencoder(input_img))
    autoencoder.compile(loss=lossFunction, optimizer=RMSprop())

    return autoencoder


def plotModelLoss(model_train, epochs):
    loss = model_train.history['loss']
    val_loss = model_train.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    plt.savefig("loss_graph_softmax_sigmoid.png")
