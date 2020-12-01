import tensorflow as tf
import numpy as np
import sys
import getopt

from util import extract_data, extract_labels, plotLoss, plotAllMetrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import backend as K

if len(sys.argv) != 11:
    print(
        "error command must look like this : python classification.py -d <training set> --d1 <training labels> -t <testset> --t1 <test labels> --model <autoencoder h5>"
    )
    sys.exit(-1)

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "d:t:", ["d1=", "t1=", "model="])
except getopt.GetoptError:
    print(
        "error command must look like this : python classification.py -d <training set> --d1 <training labels> -t <testset> --t1 <test labels> --model <autoencoder h5>"
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
        model = argument

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

train_X, valid_X, train_Y, valid_Y = train_test_split(
    train_data, train_labels, test_size=0.25, shuffle=42
)


def getParameters():
    try:
        batch_size = int(input("please enter batch_size : "))
    except ValueError:
        print("batch_size must be an integer")
        sys.exit(1)

    try:
        epochs = int(input("please enter epochs number : "))
    except ValueError:
        print("epochs must be an integer")
        sys.exit(1)

    try:
        neurons_fc_layer = int(
            input("please enter number of neurons in fc layer : "))
    except ValueError:
        print("neurons_fc_layer must be an integer")
        sys.exit(1)

    return batch_size, epochs


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


def fitModel(model, batch_size, epochs):

    output = Flatten()(autoencoder.layers[5].output)
    output = Dense(128, activation="sigmoid")(output)
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

    model_train = model.fit(
        train_X,
        train_Y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(valid_X, valid_Y),
    )

    test_loss, test_acc, test_f, test_precision, test_recall = model.evaluate(
        test_data, test_labels, verbose=2
    )
    print("\nTest accuracy:", test_acc)

    plotAllMetrics(model_train, epochs)
    return model_train


if __name__ == "__main__":

    batch_size, epochs = getParameters()

    autoencoder = load_model(model)

    model_train = fitModel(autoencoder, batch_size, epochs)

    while True:
        try:
            answer = int(
                input(
                    " press 1 if you want to repeat expiriment with different paremeters \n press 2 if you want to show plots \n press 3 if you want to categorize images \n "
                )
            )
        except ValueError:
            print("answer must be an integer")
            sys.exit(1)

        if answer == 1:
            # does it need to loead model again to not trian the same one?
            batch_size, epochs = getParameters()
            fitModel(autoencoder, batch_size, epochs)
        # elif answer == 2:
            # plotModelLoss(model_train, epochs, "models/loser.png")
        elif answer == 3:
            # ask from user which parameters he wants to use
            # categorize whatever this is
            break
        else:
            sys.exit(1)
