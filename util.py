import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


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


def plotLoss(model, name):
    # summarize history for loss
    # plt.figure()
    plt.plot(model.history["loss"])
    plt.plot(model.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(name)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plotAccuracy(model, name):
    # summarize history for accuracy
    # plt.figure()
    plt.plot(model.history["accuracy"])
    plt.plot(model.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(name)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[int(true_label)],
        ),
        color=color,
    )


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[int(true_label)].set_color("blue")
