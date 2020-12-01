import sys
import getopt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import (
    extract_data,
    extract_labels,
    plotLoss,
    plotAccuracy,
    getModel,
    plot_image,
    plot_value_array,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import backend as K

if len(sys.argv) != 11:
    print(
        "Usage: python classification.py -d <training set> --d1 <training labels> -t <testset> --t1 <test labels> --model <autoencoder h5>"
    )
    sys.exit(-1)

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "d:t:", ["d1=", "t1=", "model="])
except getopt.GetoptError:
    print(
        "Usage: python classification.py -d <training set> --d1 <training labels> -t <testset> --t1 <test labels> --model <autoencoder h5>"
    )
    sys.exit(1)
for option, argument in opts:
    if option == "-d":
        training_set = argument

    elif option in ("--d1"):
        training_labels = argument

    elif option == "-t":
        test_set = argument

    elif option in ("--t1"):
        test_labels = argument

    elif option in ("--model"):
        modelName = argument

train_data, x, y = extract_data(training_set)
test_data, x, y = extract_data(training_labels)
train_labels = extract_labels(test_set)
test_labels = extract_labels(test_labels)

inChannel = 1
train_data = train_data.reshape(-1, x, y, inChannel)
test_data = test_data.reshape(-1, x, y, inChannel)

train_data = train_data / 255.0
test_data = test_data / 255.0

train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)


def getParameters():
    try:
        batch_size = int(input("Please enter batch_size: "))
    except ValueError:
        print("batch_size must be an integer")
        sys.exit(1)

    try:
        epochs = int(input("Please enter epochs number: "))
    except ValueError:
        print("epochs must be an integer")
        sys.exit(1)

    try:
        neurons_fc_layer = int(input("Please enter number of nodes in dense layer: "))
    except ValueError:
        print("neurons_fc_layer must be an integer")
        sys.exit(1)

    return batch_size, epochs, neurons_fc_layer


def evalRecall(y_actual, y_predicted):
    return K.sum(K.round(K.clip(y_actual * y_predicted, 0, 1))) / (
        K.sum(K.round(K.clip(y_actual, 0, 1))) + K.epsilon()
    )


def evalPrecision(y_actual, y_predicted):
    return K.sum(K.round(K.clip(y_actual * y_predicted, 0, 1))) / (
        K.sum(K.round(K.clip(y_predicted, 0, 1))) + K.epsilon()
    )


def evalF(y_actual, y_predicted):
    recall = evalRecall(y_actual, y_predicted)
    precision = evalPrecision(y_actual, y_predicted)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def plotClassificationMetrics(model, name):

    precision = model.history["evalPrecision"]
    recall = model.history["evalRecall"]
    f = model.history["evalF"]

    # plt.figure()
    plt.plot(f, label="f-score")
    plt.plot(precision, label="precision")
    plt.plot(recall, label="recall")
    plt.xlabel("epochs")
    plt.legend()
    plt.show()
    plt.savefig(name)


def newModel():
    batch_size, epochs, neurons_fc_layer = getParameters()
    activationFunction = "sigmoid"

    try:
        autoencoder = load_model(modelName)
    except IOError:
        print("Model:", modelName, "doesn't exist")
        sys.exit(1)

    output = Flatten()(autoencoder.layers[5].output)
    output = Dense(neurons_fc_layer, activation=activationFunction)(output)
    output = Dense(10)(output)
    model = Model(inputs=autoencoder.input, outputs=output)

    model.summary()

    model.layers[1].trainable = False
    model.layers[3].trainable = False
    model.layers[5].trainable = False

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy", evalF, evalPrecision, evalRecall],
    )

    train_X, valid_X, train_Y, valid_Y = train_test_split(
        train_data, train_labels, test_size=0.25, shuffle=42
    )

    model_train = model.fit(
        train_X,
        train_Y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(valid_X, valid_Y),
    )

    while True:
        try:
            answer = int(
                input(
                    "Press 1 to save the model\nPress 2 to print training plots\nPress 3 to show predictions\n"
                )
            )
        except ValueError:
            print("Answer must be an integer")

        if answer == 1:
            model.save(str(input("Provide the name of the file: ")))
        elif answer == 2:
            plotLoss(model_train, "model_loss.png")
            plotAccuracy(model_train, "model_accuracy")
            plotClassificationMetrics(model_train, "model_metrics.png")
        elif answer == 3:
            break
        else:
            print("Invalid input")

    return model


def modelPredict(model, test_data, test_labels):
    predictions = model.predict(test_data)

    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_data)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
        plt.tight_layout()
    plt.savefig("prediction.png")
    plt.show()


if __name__ == "__main__":

    while True:
        try:
            answer = int(
                input(
                    "Press 1 to train a new model\nPress 2 to load an existing model\nPress 3 to exit\n"
                )
            )
        except ValueError:
            print("Answer must be an integer")

        if answer == 1:  # New Model
            model = newModel()
            modelPredict(model, test_data, test_labels)
        elif answer == 2:  # Load Model
            model = getModel()
            modelPredict(model, test_data, test_labels)
        elif answer == 3:
            break
        else:
            print("Invalid input")
