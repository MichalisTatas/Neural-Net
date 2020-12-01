import numpy as np
import matplotlib.pyplot as plt


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


def extract_labels(filename):

    with open(filename, "rb") as f:
        f.read(4)
        num_images = int.from_bytes(f.read(4), "big", signed=True)
        buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
        labels = labels.reshape(num_images).astype(np.float32)
        f.close()

        return labels


def plotModelLoss(model_train, epochs, name):
    loss = model_train.history["loss"]
    val_loss = model_train.history["val_loss"]
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    # plt.show()
    plt.savefig(name)

def plotAllMetrics(model, epochs):

    epochs = range(epochs)

    loss = model.history["loss"]
    accuracy = model.history["accuracy"]
    precision = model.history["evalPrecision"]
    recall = model.history["evalRecall"]
    f = model.history["evalF"]
    plt.plot(epochs, loss, label="loss")
    plt.plot(epochs, accuracy, label="accuracy")

    plt.xlabel('loss')
    plt.ylabel('metrics')
    # plt.legend(loc='best')
    # plt.savefig('images/emiris.png')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, f, label="f")
    plt.plot(epochs, precision, label="precision")
    plt.plot(epochs, recall, label="recall")
    plt.legend()
    plt.show()