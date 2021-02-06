import tensorflow.keras as keras
from keras.layers import Dense, InputLayer


def calc_width(initial_w, l):
    """ Calculate the width for all hidden layers.

    :param initial_w: width of first layer
    :param l: number of hidden layers
    """
    widths = [initial_w]
    for i in range(1, l):
        widths.append(int(widths[i-1] / 2))

    return widths


def get_model(x_train, y_train, w1, lmax):
    """ Create, compile and train the mlp model.

    :param x_train: feature training data
    :param y_train: label training data
    :param w1: width of first hidden layer
    :param lmax: number of hidden layers
    """
    input_shape = (len(x_train[0]),)
    widths = calc_width(w1, lmax)

    model = keras.Sequential()
    # add input layer
    model.add(InputLayer(input_shape))
    # add hidden layers
    for i in range(len(widths)):
        model.add(Dense(widths[i], activation='relu'))
    # add output layer, also tried this out with activation='relu' but result for unnormalized dataset became way worse
    model.add(Dense(1))
    # compile and fit model
    model.compile(optimizer='adam', loss='mse')
    # fit model using the training data
    model.fit(x=x_train, y=y_train, batch_size=32, epochs=100, verbose=0, shuffle=True)

    return model


def normalize_data(x_train, x_validate):
    """ Min-Max normalize the feature data.

    :param x_train: feature training data
    :param x_validate: feature validation data
    :return: x_train_norm, x_validate_norm
    """
    min_x = x_train.min(axis=0)
    max_x = x_train.max(axis=0)
    x_train = (x_train - min_x) / (max_x - min_x)
    x_validate = (x_validate - min_x) / (max_x - min_x)

    return x_train, x_validate


def grid_search(x_train, y_train, x_validate, y_validate):
    """ Perform a grid search to find optimal values for w1 and lmax.

    :param x_train: feature training data
    :param y_train: label training data
    :param x_validate: feature validation data
    :param y_validate: label validation data
    :return: ((w1_best, lmax_best), (mse))
    """
    # define the grid
    w_values = [1024, 512, 256, 128, 64, 32]
    l_values = [2, 3, 4, 5, 6, 7]

    # store trained models in here
    models = {}

    # grid search
    for w in w_values:
        for l in l_values:
            keras.backend.clear_session()
            # get trained model
            model = get_model(x_train, y_train, w, l)
            # evaluate and store params and score in models dictionary
            models[(w, l)] = model.evaluate(x=x_validate, y=y_validate)

    # print params and score of all models
    for k, v in models.items():
        print("{}: {}".format(k, v))

    # find best model
    best = min(models.items(), key=lambda x: x[1])

    return best


def main():
    # load data
    (x_train, y_train), (x_validate, y_validate) = keras.datasets.boston_housing.load_data()
    # initialize params
    w1 = 512
    lmax = 2

    print("---------- Unnormalized dataset ----------")
    model = get_model(x_train, y_train, w1, lmax)
    print("Evaluation:")
    model.evaluate(x=x_validate, y=y_validate)

    print("---------- Normalized dataset ----------")
    # normalize data
    x_train_norm, x_validate_norm = normalize_data(x_train, x_validate)
    model_ = get_model(x_train_norm, y_train, w1, lmax)
    print("Evaluation:")
    model_.evaluate(x=x_validate_norm, y=y_validate)

    print("---------- Grid search on normalized dataset ----------")
    # for testing purposes the grid search was iterated 5 times, set this to 1 to not iterate the grid search
    iterations = 5
    best_list = []
    for i in range(iterations):
        best_list.append(grid_search(x_train_norm, y_train, x_validate_norm, y_validate))
    for i in range(iterations):
        print("Best parameters were w1 = {}, lmax = {}. MSE of model was {}".format(best_list[i][0][0],
                                                                                    best_list[i][0][1],
                                                                                    best_list[i][1]))

    """ normalized data results:
    Best parameters were w1 = 1024, lmax = 6. MSE of model was 15.374990463256836
    Best parameters were w1 = 1024, lmax = 7. MSE of model was 13.135924339294434
    Best parameters were w1 = 1024, lmax = 3. MSE of model was 12.833697319030762
    Best parameters were w1 = 1024, lmax = 4. MSE of model was 11.404892921447754
    Best parameters were w1 = 1024, lmax = 6. MSE of model was 12.431068420410156
    """

    """ unnormalized data results: (this was made for comparison)
    Best parameters were w1 = 256, lmax = 2. MSE of model was 24.796497344970703
    Best parameters were w1 = 128, lmax = 4. MSE of model was 26.780900955200195
    Best parameters were w1 = 128, lmax = 3. MSE of model was 24.112680435180664
    Best parameters were w1 = 256, lmax = 2. MSE of model was 25.49637222290039
    Best parameters were w1 = 256, lmax = 3. MSE of model was 24.930578231811523
    """


if __name__ == '__main__':
    main()
