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


def plotLoss(model, name):
    # summarize history for loss
    plt.figure()
    plt.plot(model.history["loss"])
    plt.plot(model.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
    plt.savefig(name)


def plotAccuracy(model, name):
    # summarize history for accuracy
    plt.figure()
    plt.plot(model.history["accuracy"])
    plt.plot(model.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
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

    plt.xlabel("loss")
    plt.ylabel("metrics")
    # plt.legend(loc='best')
    plt.legend()
    plt.savefig("images/emiris.png")
    plt.show()

    plt.figure()
    plt.plot(epochs, f, label="f")
    plt.plot(epochs, precision, label="precision")
    plt.plot(epochs, recall, label="recall")
    plt.legend()
    plt.savefig("images/emiris2.png")
    plt.show()
