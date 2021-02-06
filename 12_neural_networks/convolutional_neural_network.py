import numpy as np
import tensorflow.keras as keras

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

from timeit import default_timer as timer


def get_model(input_shape):
    """ Get the untrained CNN model.

    :param input_shape: shape of input data
    :return: keras Sequential model
    """
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu'),     # convolutional layer 1
        Conv2D(filters=64, kernel_size=3, activation='relu'),                              # convolutional layer 2
        MaxPooling2D(pool_size=2),                                                         # pooling layer
        Flatten(),                                                                         # flatten matrix input
        Dense(32, activation='relu'),                                                      # dense layer
        Dense(10, activation='softmax')                                                    # output layer
    ])
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model


def test_model(input_shape):
    """ Get an untrained CNN model. This model has some different hyper parameters.

    :param input_shape: shape of input data
    :return: keras Sequential model
    """
    model = Sequential([
        Conv2D(filters=32, kernel_size=3, input_shape=input_shape, activation='relu'),
        Conv2D(filters=32, kernel_size=3, activation='relu'),
        Conv2D(filters=32, kernel_size=5, activation='relu', strides=2),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        Conv2D(filters=64, kernel_size=3, activation='relu'),
        Conv2D(filters=64, kernel_size=5, activation='relu', strides=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model


def get_data():
    """ Load mnist dataset, scale and reshape the data.

    :return (x_train, y_train), (x_validate, y_validate)
    """
    # load data
    (x_train, y_train), (x_validate, y_validate) = keras.datasets.mnist.load_data()

    # data has values from 0-255 -> divide by 255 to get values from 0 to 1
    x_train = x_train / 255
    x_validate = x_validate / 255

    # reshape data by adding a dimension
    x_train = np.expand_dims(x_train, axis=3)
    x_validate = np.expand_dims(x_validate, axis=3)

    # turn labels to vector
    y_train = to_categorical(y_train)
    y_validate = to_categorical(y_validate)

    return (x_train, y_train), (x_validate, y_validate)


def main():
    # get training and validation data
    (x_train, y_train), (x_validate, y_validate) = get_data()

    # get and train the first model
    start1 = timer()
    model = get_model(x_train[0].shape)
    model.fit(x_train, y_train, epochs=12, batch_size=32, shuffle=True)
    end1 = timer()

    # get and train the test model
    start2 = timer()
    test = test_model(x_train[0].shape)
    test.fit(x_train, y_train, epochs=12, batch_size=32, shuffle=True)
    end2 = timer()

    print("\n----------------- Model evaluation -----------------")
    print("First model: ")
    model.evaluate(x_validate, y_validate)
    print("Second model: ")
    test.evaluate(x_validate, y_validate)

    print("Training of first model done after {}s.".format(end1 - start1))
    print("Training of second model done after {}s.".format(end2 - start2))

    """
    First model: loss: 0.0607 - accuracy: 0.9880
    Second model: loss: 0.0341 - accuracy: 0.9923
    Training of first model done after 475.1620607s.
    Training of second model done after 587.2978638s.
    """


if __name__ == '__main__':
    main()
