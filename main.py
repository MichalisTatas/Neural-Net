import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from autoencoder import getAutoencoder, plotModelLoss


def extract_data(filename):
    with open(filename, "rb") as f:
        f.read(4)
        num_images = int.from_bytes(f.read(4), "big", signed=True)
        x = int.from_bytes(f.read(4), "big", signed=True)
        y = int.from_bytes(f.read(4), "big", signed=True)
        buf = f.read(num_images * x * y)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, x, y).astype(np.float32)
        f.close()

        return data, x, y


data, x, y = extract_data("./data/t10k-images-idx3-ubyte")

data = data.reshape(-1, x, y, 1)  # img_num * x * y * 1
data = data / np.max(data)

train_X, valid_X, train_ground, valid_ground = train_test_split(
    data, data, test_size=0.25, shuffle=42)

batch_size = 128
epochs = 50
# autoencoder = getAutoencoder(
#     x=x, y=y, activationFunction="softmax", lastActivationFunction="sigmoid", lossFunction="mean_squared_error")
# autoencoder.summary()
# autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,
#                                     epochs=epochs, verbose=2, validation_data=(valid_X, valid_ground))
# autoencoder.save("models/autoencoder_softmax_sigmoid")
autoencoder = load_model("models/autoencoder_softmax_sigmoid")

results = autoencoder.evaluate(data, data, batch_size=128)
print("test loss, test acc:", results)

# ploting loss graph
# plotModelLoss(autoencoder_train, epochs, "models/model_softmax_sigmoid/plot.png")

# pred = autoencoder.predict(data)
# plt.figure(figsize=(20, 4))
# print("Test Images")
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(data[i, ..., 0], cmap='gray')
# plt.show()
# plt.savefig("original.png")
# plt.figure(figsize=(20, 4))
# print("Reconstruction of Test Images")
# for i in range(10):
#     plt.subplot(2, 10, i+1)
#     plt.imshow(pred[i, ..., 0], cmap='gray')
# plt.show()
# plt.savefig("result.png")
