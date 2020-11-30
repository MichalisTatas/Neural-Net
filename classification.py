import tensorflow as tf
import numpy as np
import sys
import getopt

from util import extract_data, extract_labels, plotModelLoss
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Accuracy

argv = sys.argv[1:]

if len(sys.argv) != 11:
    print ("error command must look like this : python classification.py -d <training set> -d1 <training labels> -t <testset> -t1 <test labels> -model <autoencoder h5>")
    sys.exit(-1)

try:
    opts, args = getopt.getopt(argv,"d:t:", ["d1=","t1=","model="])
except getopt.GetoptError:
        print ("error command must look like this : python classification.py -d <training set> -d1 <training labels> -t <testset> -t1 <test labels> -model <autoencoder h5>")
        sys.exit(1)
for option, argument in opts:
    if   option == '-d':
        training_set = argument

    elif option in ("--d1"):
        training_labels = argument

    elif option == '-t':
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

train_data = train_data / np.max(train_data)
test_data = test_data / np.max(test_data)

train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

train_X, valid_X, train_Y, valid_Y = train_test_split(
    train_data, train_labels, test_size=0.25, shuffle=42
)

autoencoder = load_model(model)

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
    metrics=["accuracy"],
)

batch_size = 128
epochs = 8
model_train = model.fit(
    train_X,
    train_Y,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_Y),
)

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)


plotModelLoss(model_train, epochs, "models/loser.png")
