import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def split_data(df):
    """ Takes a dataframe, splits it up in feature and label data and returns that data split into
    training and validation data.

    :param df: pandas dataframe
    :return: x_training, x_validation, y_training, y_validation
    """
    feature_data, label_data = split_feature_label_data(df)

    # don't shuffle data to make results reproducible
    return train_test_split(feature_data, label_data, test_size=0.2, random_state=123)


def split_feature_label_data(df):
    """ Takes a dataframe and splits it into two, one being the feature data and one the label data.

    :param df: pandas dataframe
    :return: feature df, label df
    """
    feature_data = df.loc[:, df.columns != 'disease']
    label_data = df.loc[:, df.columns == 'disease']

    return feature_data, label_data


def print_mse(model, x_training, x_validation, y_training, y_validation):
    """ Print out the mse of the training data and the mse of the validation data for the given model.

    :param model: Regressor model
    :param x_training: Feature data for training
    :param x_validation: Feature data for validation
    :param y_training: Label data for training
    :param y_validation: Label data for validation
    """
    model.fit(x_training, y_training)

    mse_training = mean_squared_error(y_training, model.predict(x_training))
    mse_validation = mean_squared_error(y_validation, model.predict(x_validation))

    print('MSE of training data: {}'.format(mse_training))
    print('MSE of validation data: {}'.format(mse_validation))


def task_1_2(x_training, x_validation, y_training, y_validation):
    """ Get MSE according to step 1 and 2 of the task.

    :param x_training: Feature data for training
    :param x_validation: Feature data for validation
    :param y_training: Label data for training
    :param y_validation: Label data for validation
    """
    ols = LinearRegression()
    print('------------ MSE OLS ------------')
    print_mse(ols, x_training, x_validation, y_training, y_validation)

    lasso = Lasso(alpha=0.1)
    print('------------ MSE Lasso------------')
    print_mse(lasso, x_training, x_validation, y_training, y_validation)


def get_pairs_from_correlations(correlating_features):
    """ Get a list of pairs of correlating features.

    :param correlating_features: list of indices from the correlation matrix
    :return list of tuples
    """
    pairs = []
    for i in range(len(correlating_features[0])):
        pairs.append((correlating_features[0][i], correlating_features[1][i]))
    del pairs[::2]

    return pairs


def get_unique_features(pairs):
    """ Get a list of each feature that is present in any pair.

    :param pairs: list of tuples
    :return: list
    """
    unique_features = []
    for i in range(len(pairs)):
        if pairs[i][0] not in unique_features:
            unique_features.append(pairs[i][0])
        if pairs[i][1] not in unique_features:
            unique_features.append(pairs[i][1])
    return sorted(unique_features)


def count_appearances(pairs):
    """ Count the appearance of each value in pairs and return those counts in a dictionary.

    :param pairs: list of tuples
    :return: dictionary
    """
    unique_features = get_unique_features(pairs)
    dic = {f: 0 for f in unique_features}
    for pair in pairs:
        dic[pair[0]] += 1
        dic[pair[1]] += 1

    return dic


def correlation_based_feature_selection(df, threshold):
    """ Reduce feature space with correlation based feature selection.

    :param df: pandas dataframe
    :param threshold: correlation threshold
    :return: pandas dataframe with fewer features
    """
    # instead of calc_correlation maybe use np.corrcoef
    correlation_matrix = np.zeros((df.shape[1], df.shape[1]))
    for i in range(len(df.columns)):
        for j in range(len(df.columns)):
            correlation_matrix[i][j] = abs(np.corrcoef(df[df.columns[i]], df[df.columns[j]])[0][1])
    # fill diagonals with 0 cause else they would be 1 and add some extra unwanted pairs
    np.fill_diagonal(correlation_matrix, 0)
    # find correlating features
    correlating_features = np.where(correlation_matrix > threshold)
    # get pairs of correlating features
    pairs = get_pairs_from_correlations(correlating_features)
    # store removed features to drop later
    removed_features = []
    # while correlating features exist
    while len(pairs) > 0:
        # count appearances of features and remove the one with the most appearances
        pair_count = count_appearances(pairs)
        max_count = max(pair_count, key=lambda feature: pair_count[feature])
        remove_feature = max_count
        # remove all pairs with that feature
        pairs = [pair for pair in pairs if remove_feature not in pair]
        removed_features.append(df.columns[remove_feature])

    return df.drop(removed_features, axis=1)


def pca(df, p):
    """ Reduce feature space with principal component analysis

    :param df: pandas dataframe
    :param p: dimension of new feature space
    :return: pandas dataframe with fewer features
    """
    x = df
    y = np.transpose(x) @ x
    eigvec = np.linalg.eig(y)[1]
    w = eigvec
    wp = w[:, :p]
    tp = x @ wp

    return tp


def main():
    file_path = 'diabetes.csv'
    df = pd.read_csv(file_path)
    x_training, x_validation, y_training, y_validation = split_data(df)

    task_1_2(x_training, x_validation, y_training, y_validation)

    print('------------ CBFS results ------------')
    cbfs_feature_space = correlation_based_feature_selection(split_feature_label_data(df)[0], 0.6)
    print('Columns: {}'.format(cbfs_feature_space.columns))
    x_training_cbfs, x_validation_cbfs, y_training_cbfs, y_validation_cbfs = \
        train_test_split(cbfs_feature_space, split_feature_label_data(df)[1], test_size=0.2, random_state=123)
    print_mse(LinearRegression(), x_training_cbfs, x_validation_cbfs, y_training_cbfs, y_validation_cbfs)

    pca_feature_space = pca(split_feature_label_data(df)[0], 2)
    x_training_pca, x_validation_pca, y_training_pca, y_validation_pca = \
        train_test_split(pca_feature_space, split_feature_label_data(df)[1], test_size=0.2, random_state=123)
    print('------------ PCA results ------------')
    print_mse(LinearRegression(), x_training_pca, x_validation_pca, y_training_pca, y_validation_pca)


if __name__ == '__main__':
    main()
