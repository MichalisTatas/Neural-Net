import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D

f = open("./data/t10k-images-idx3-ubyte", "rb")

f.read(4)
data_n = int.from_bytes(f.read(4), "big", signed=True)
x = int.from_bytes(f.read(4), "big", signed=True)
y = int.from_bytes(f.read(4), "big", signed=True)

data = []
for i in range(0, data_n):
    point = []

    for j in range(0, x*y):
        byte = f.read(1)

        if byte != b"":
            point.append(int.from_bytes(byte, "little"))

    data.append(point)
f.close()

data = np.array(data, dtype=np.uint8)
print(data.shape)

train, test = train_test_split(data)
print(train.shape)
print(test.shape)


def encoder(image):
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(image)

    return conv1


encoder(train[0])
