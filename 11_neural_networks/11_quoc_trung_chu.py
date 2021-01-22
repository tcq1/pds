import tensorflow.keras as keras
import numpy as np


def calc_width(initial_w, l):
    """ Calculate the width for all hidden layers.

    :param initial_w: width of first layer
    :param l: number of hidden layers
    """
    widths = [initial_w]
    for i in range(1, l):
        widths.append(int(widths[i-1] / 2))

    return widths


def get_model(data, w1, lmax):
    """ Create, compile and train the mlp model.

    :param data: input data from k.datasets.boston_housing.load_data()
    :param w1: width of first hidden layer
    :param lmax: number of hidden layers
    """
    input_shape = (len(data[0][0][0]),)
    widths = calc_width(w1, lmax)

    model = keras.Sequential()
    # add input layer
    model.add(keras.layers.InputLayer(input_shape))
    # add hidden layers
    for i in range(1, len(widths)):
        model.add(keras.layers.Dense(widths[i], activation='relu'))
    # compile and fit model
    model.compile(optimizer='adam', loss='mse')
    # fit model using the training data
    model.fit(x=data[0][0], y=data[0][1], batch_size=32, epochs=100, verbose=0)

    return model


def normalize_data(data):
    """ Min-Max normalize the feature data.

    :param data: input data from k.datasets.boston_housing.load_data()
    :return: normalized data
    """
    # iterate over training and test data
    for i in range(len(data)):
        # get min and max values of columns
        min_x = data[i][0].min(axis=0)
        max_x = data[i][0].max(axis=0)
        # iterate over columns
        for j in range(len(data[i][0].T)):
            # normalize all values of the column
            for k in range(len(data[i][0].T[j])):
                # update value
                data[i][0].T[j][k] = (data[i][0].T[j][k] - min_x[j]) / (max_x[j] - min_x[j])

    return data


def grid_search(data):
    """ Perform a grid search to find optimal values for w1 and lmax.

    :param data: normalized data
    :return: ((w1_best, lmax_best), (mse, mse))
    """
    # define the grid
    w_values = [1024, 512, 256, 128, 64, 32]
    l_values = [2, 3, 4, 5, 6, 7]

    models = {}

    for w in w_values:
        for l in l_values:
            model = get_model(data, w, l)
            models[(w, l)] = model.evaluate(x=data[1][0], y=data[1][1])

    for k, v in models.items():
        print("{}: {}".format(k, v))

    best = min(models.items(), key=lambda x: x[1])

    return best


def main():
    data = keras.datasets.boston_housing.load_data()
    w1 = 512
    lmax = 2

    print("---------- Unnormalized dataset ----------")
    model = get_model(data, w1, lmax)
    print("Evaluation:")
    model.evaluate(x=data[1][0], y=data[1][1])

    print("---------- Normalized dataset ----------")
    print("Normalizing data...")
    normalized = normalize_data(data)
    model_ = get_model(normalized, w1, lmax)
    print("Evaluation:")
    model_.evaluate(x=normalized[1][0], y=normalized[1][1])

    print("---------- Grid search on normalized dataset ----------")
    best_list = []
    for i in range(5):
        best_list.append(grid_search(normalized))
    for i in range(5):
        print("Best parameters were w1 = {}, lmax = {}. MSE of model was {}".format(best_list[i][0][0],
                                                                                    best_list[i][0][1],
                                                                                    best_list[i][1]))

    """ normalized data results:
    Best parameters were w1 = 64, lmax = 7. MSE of model was 38.789188385009766
    Best parameters were w1 = 64, lmax = 6. MSE of model was 34.2607421875
    Best parameters were w1 = 32, lmax = 5. MSE of model was 39.406246185302734
    Best parameters were w1 = 64, lmax = 6. MSE of model was 36.057586669921875
    Best parameters were w1 = 128, lmax = 5. MSE of model was 39.83441162109375
    """

    """ unnormalized data results: (this was made for comparison)
    Best parameters were w1 = 64, lmax = 6. MSE of model was 38.22695541381836
    Best parameters were w1 = 32, lmax = 6. MSE of model was 38.06871032714844
    Best parameters were w1 = 64, lmax = 5. MSE of model was 37.287628173828125
    Best parameters were w1 = 128, lmax = 7. MSE of model was 36.05194091796875
    Best parameters were w1 = 128, lmax = 5. MSE of model was 38.385101318359375
    """


def test():
    a_label = np.random.randint(0, 10, size=5)
    a_feature = np.random.random(3)*10
    b_label = np.random.randint(0, 10, size=5)
    b_feature = np.random.random(3)*10
    for i in range(4):
        a_feature = np.vstack([a_feature, np.random.random(3)*10])
        b_feature = np.vstack([b_feature, np.random.random(3)*10])
    data = ((a_feature, a_label), (b_feature, b_label))
    print(data)
    print("Normalized:")
    print(normalize_data(data))


if __name__ == '__main__':
    main()
