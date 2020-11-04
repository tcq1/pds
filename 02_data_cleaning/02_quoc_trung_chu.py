import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_bounds_iqr(data):
    """ Returns outlier bounds of a given numpy array
    """

    temperatures = data[:, 1]

    q25 = np.percentile(temperatures, 25)
    q75 = np.percentile(temperatures, 75)
    iqr = q75 - q25

    lower = q25 - (1.5 * iqr)
    upper = q75 + (1.5 * iqr)

    print('Q25 = {}, Q75 = {}, IQR = {}'.format(q25, q75, iqr))
    print('IQR Filter: Lower bound = {}, Upper bound = {}'.format(lower, upper))

    return [lower, upper]


def iqr_filter(data):
    """ Uses IQR filter to remove outliers

    :param data: numpy array
    :return: numpy array
    """

    bounds = get_bounds_iqr(data)
    indices_to_remove = remove_outliers(data, bounds)
    for x in indices_to_remove:
        print('Removed {} with IQR filter'.format(data[x]))
    print('Removed {} data points with IQR filter'.format(len(indices_to_remove)))

    return replace_with_nan(data, indices_to_remove)


def z_score_filter(data):
    """ Uses Z-Score filter to remove outliers

    :param data: numpy array
    :return: numpy array
    """

    normalized = data.copy()
    mean = np.mean(data[:, 1], axis=0)
    sd = np.std(data[:, 1], axis=0)
    print('Mean = {}, Standard deviation = {}'.format(mean, sd))
    for i in range(len(normalized)):
        normalized[i][1] = (normalized[i][1] - mean) / sd

    indices_to_remove = remove_outliers(normalized, [-3, 3])
    for x in indices_to_remove:
        print('Removed {} with Z-score-filter'.format(data[x]))
    print('Removed {} data points with Z-score-filter'.format(len(indices_to_remove)))

    return replace_with_nan(data, indices_to_remove)


def remove_outliers(data, bounds):
    """ Returns indices of outliers in a numpy array

    :param data: numpy array
    :param bounds: bounds, in which numpy array values are accepted
    :return: list of indices that need to be removed
    """
    lower, upper = bounds
    indices_to_delete = []

    for i in range(len(data)):
        if not lower < data[i][1] < upper:
            indices_to_delete.append(i)

    return indices_to_delete


def replace_with_nan(data, indices_to_delete):
    """ Replaces values of specified indices with nan

    :param data: numpy array
    :param indices_to_delete: list
    :return: numpy array
    """

    print(indices_to_delete)
    for x in indices_to_delete:
        data[x][1] = np.nan

    return data


def main():
    # read csv and save as numpy array
    df = pd.read_csv('data-cleaning.csv')
    data = df.to_numpy()

    # get filtered data
    filtered_data = iqr_filter(data)
    filtered_data_2 = z_score_filter(data)

    # plotting
    ax = plt.gca()
    columns = ['timestamp', 'temperature']

    df.plot(kind='line', y='temperature', ax=ax, label='Given data')
    pd.DataFrame(data=filtered_data, columns=columns).plot(kind='line', y='temperature', ax=ax, label='IQR')
    pd.DataFrame(data=filtered_data_2, columns=columns).plot(kind='line', y='temperature', ax=ax, label='Z-score')
    plt.show()


if __name__ == '__main__':
    main()
