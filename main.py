import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop


def extract_data(filename):
    with open(filename, "rb") as file:
        file.read(4)
        num_images = int.from_bytes(file.read(4), "big", signed=True)
        x = int.from_bytes(file.read(4), "big", signed=True)
        y = int.from_bytes(file.read(4), "big", signed=True)
        buf = file.read(num_images * x * y)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, x, y).astype(np.float32)

        return data, x, y


data, x, y = extract_data("./data/t10k-images-idx3-ubyte")

print(data.shape)
data = data.reshape(-1, x, y, 1)  # img_num * x * y * 1
print(data.shape)
print(data.dtype)

data = data / np.max(data)

train_X, valid_X, train_ground, valid_ground = train_test_split(
    data, data, test_size=0.25, shuffle=42)


def encoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu',
                   padding='same')(input_img)  # 28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32

    conv2 = Conv2D(64, (3, 3), activation='relu',
                   padding='same')(pool1)  # 14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64

    conv3 = Conv2D(128, (3, 3), activation='relu',
                   padding='same')(pool2)  # 7 x 7 x 128

    return conv3


def decoder(input_conv):
    conv3 = Conv2D(128, (3, 3), activation='relu',
                   padding='same')(input_conv)  # 7 x 7 x 128
    up3 = UpSampling2D((2, 2))(conv3)  # 14 x 14 x 128

    conv2 = Conv2D(64, (3, 3), activation='relu',
                   padding='same')(up3)  # 14 x 14 x 64
    up2 = UpSampling2D((2, 2))(conv2)  # 28 x 28 x 64

    covn1 = Conv2D(1, (3, 3), activation='sigmoid',
                   padding='same')(up2)  # 28 x 28 x 1

    return covn1


def autoencoder(input_img):
    return decoder(encoder(input_img))


batch_size = 128
epochs = 50
inChannel = 1
lossFunction = "mean_squared_error"
input_img = Input(shape=(x, y, inChannel))

autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss=lossFunction, optimizer=RMSprop())
print(autoencoder.summary())

autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,
                                    epochs=epochs, verbose=1, validation_data=(valid_X, valid_ground))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# pred = autoencoder.predict(test_data)
# plt.figure(figsize=(20, 4))
# print("Test Images")
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(test_data[i, ..., 0], cmap='gray')
#     curr_lbl = test_labels[i]
#     plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
# plt.show()
# plt.figure(figsize=(20, 4))
# print("Reconstruction of Test Images")
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(pred[i, ..., 0], cmap='gray')
# plt.show()
