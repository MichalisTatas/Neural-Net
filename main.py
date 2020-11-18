import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, BatchNormalization


def extract_data(filename):
    with open(filename, "rb") as file:
        file.read(4)
        num_images = int.from_bytes(file.read(4), "big", signed=True)
        x = int.from_bytes(file.read(4), "big", signed=True)
        y = int.from_bytes(file.read(4), "big", signed=True)
        buf = file.read(num_images * x * y)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, x, y)

        return data


data = extract_data("./data/t10k-images-idx3-ubyte")

# Display the first image in training data
plt.figure(figsize=[5, 5])
plt.subplot(121)
curr_img = np.reshape(data[0], (28, 28))
plt.imshow(curr_img, cmap='gray')
plt.show()


train, test = train_test_split(data)
print(train.shape)
print(test.shape)
