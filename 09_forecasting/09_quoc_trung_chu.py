import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error as mse


def get_time_series(df):
    """ Get a list of all time series of the given data.

    :param df: Dataframe containing the time series
    :return: list of dataframes with time series
    """
    time_series = []
    # get shop ids
    shop_ids = df.shop_id.unique()

    # get time series for each shop and append to time_series
    for shop in shop_ids:
        series = df[df['shop_id'] == shop]
        time_series.append(series)

    return time_series


def ar(series, params, c, offset):
    """ Calculate the auto regression part.

    :param series: dataframe of the time series
    :param params: list of coefficients
    :param c: constant
    :param offset: index of last predicted value
    :return: float
    """
    result = c
    for i in range(len(params)):
        result += params[i] * series.iloc[offset-i]['transactions']

    return result


def ma(series_valid, series_predicted, params, offset):
    """ Calculate the moving average part.

    :param series_valid: dataframe of the full time series
    :param series_predicted: dataframe of predicted values
    :param params: list of coefficients
    :param offset: index of last predicted value
    :return: float
    """
    result = 0

    for i in range(len(params)):
        # calculate error
        if offset - i < len(params):
            error = 0
        else:
            error = series_valid.iloc[offset-i]['transactions'] - series_predicted.iloc[offset-i]['transactions']

        result += params[i] * error

    return result


def arma(series, p_values, c, q_values):
    """ ARMA forecasting model.

    :param series: dataframe of the full time series
    :param p_values: parameters for AR part of length p
    :param c: constant for AR part
    :param q_values: parameters for MA part of length q
    """
    predicted = pd.DataFrame(columns=series.columns)

    # predict every value of given series
    for i in range(len(series)):
        # don't predict first values
        if i < min([len(p_values), len(q_values)]):
            transactions_value = np.nan
        else:
            transactions_value = ar(series, p_values, c, i-1) + ma(series, predicted, q_values, i-1)
        # add to predicted df
        predicted = predicted.append({'shop_id': series.loc[i]['shop_id'],
                                      'time': series.loc[i]['time'],
                                      'transactions': transactions_value},
                                     ignore_index=True)

    return predicted


def get_mse(true, predicted):
    """ Get the mse, skipping nan values.

    :param true: dataframe with true values
    :param predicted: dataframe with predicted values
    """
    # get rows with nan values
    nan_rows = predicted.loc[pd.isna(predicted['transactions']), :].index

    return mse(true.drop(nan_rows), predicted.drop(nan_rows))


def main():
    file_path = 'ts_data.csv'
    df = pd.read_csv(file_path)
    time_series = get_time_series(df)
    predicted = arma(time_series[0], [0.7, 0.3], 0, [0.5, 0.5])
    print(get_mse(time_series[0], predicted))

    plt.plot(time_series[0]['time'], time_series[0]['transactions'], predicted['time'], predicted['transactions'])
    plt.xlabel('Time')
    plt.ylabel('Transactions')
    plt.show()


if __name__ == '__main__':
    main()
