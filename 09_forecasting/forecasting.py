import numpy as np
import pandas as pd

from timeit import default_timer as timer
from scipy.optimize import minimize
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
        # also reset the indexing to avoid errors later on
        series = df[df['shop_id'] == shop].reset_index(drop=True)
        time_series.append(series)

    return time_series


def ar(series, params, offset):
    """ Calculate the auto regression part.

    :param series: list with transactions
    :param params: list of coefficients, last value is the constant
    :param offset: index of last predicted value
    :return: float
    """
    return params[-1] + sum([params[i] * series[offset-i] for i in range(len(params) - 1)])


def ma(error, params, offset):
    """ Calculate the moving average part.

    :param error: list of error terms
    :param params: list of coefficients
    :param offset: index of last predicted value
    :return: float
    """
    return sum([params[i] * error[offset - i] for i in range(len(params))])


def arma(series, params, p, q):
    """ ARMA forecasting model.

    :param series: list of transactions
    :param params: list of parameters of structure [c p_0, ..., p_p, constant, q_0, ..., q_q]
    :param p: number of parameters for the AR part (excluding constant)
    :param q: numbers of parameters for the MA part
    """
    predicted = []
    error = []

    p_params = params[:p+1]
    q_params = params[p+1:]

    # predict every value of given series
    for i in range(len(series)):
        # calculate ar and ma part
        ar_part = ar(series, p_params, i-1) if i >= p and i > 0 else np.nan
        ma_part = ma(error, q_params, i-1) if i >= q and i > 0 else np.nan

        # if both are still nan then set value to nan, else get np.nansum of both parts
        if np.isnan(ar_part) and np.isnan(ma_part):
            transactions_value = np.nan
        else:
            transactions_value = np.nansum([ar_part, ma_part])
        # calculate error
        error.append(series[i] - transactions_value if i >= q else 0)

        # add to predicted df
        predicted.append(transactions_value)

    return predicted


def get_mse(true, predicted):
    """ Get the mse, skipping nan values.

    :param true: list with true values
    :param predicted: list with predicted values
    """
    # skip all nan values from mse calculation
    i = 0
    if np.nan in predicted:
        while np.isnan(predicted[i]):
            i += 1

    return mse(true[i:], predicted[i:])


def grid_search(series, p_dim, q_dim):
    """ Grid search to find optimal number of parameters for the model.

    :param series: list of transaction values
    :param p_dim: maximum number of parameters for the AR part
    :param q_dim: maximum number of parameters for the MA part
    """
    # store scores of different models
    scores = {}

    # grid search: try out all combinations for p and q
    for p in range(p_dim + 1):
        for q in range(q_dim + 1):
            # nelder mead optimizer
            score = minimize(optimize_function, x0=np.zeros(p+1+q), args=(series, p, q), method='Nelder-Mead')
            # add score to dict
            try:
                if scores[(p, q)].fun < score.fun:
                    scores[(p, q)] = score
            except KeyError:
                scores[(p, q)] = score

    # get best model
    best = min(scores.items(), key=lambda x: x[1].fun)

    return best


def optimize_function(params, series=None, p=1, q=1):
    """ Combine the ARMA model with a MSE score to optimize.
    This function should be used in the optimizer.

    :param params: list of parameters of structure [p_0, ..., p_p, constant, q_0, ..., q_q]
    :param series: list with transaction values
    :param p: number of parameters for the AR part
    :param q: number of parameters for the MA part
    :return: MSE
    """
    prediction = arma(series, params, p, q)

    return get_mse(series, prediction)


def main():
    file_path = 'ts_data.csv'
    df = pd.read_csv(file_path)
    time_series = get_time_series(df)

    start_total = timer()
    for shop in time_series:
        start_shop = timer()
        best = grid_search(shop['transactions'], 3, 3)
        print("Done with shop {} after {}s.".format(shop.iloc[0]['shop_id'], timer() - start_shop))
        print("p: {}, q: {}, MSE: {}, parameters: {}\n".format(best[0][0], best[0][1], best[1].fun, best[1].x))

    print("Done with all time series after {}s.".format(timer() - start_total))

    """ Result:
    
    Done with shop 145 after 28.5056102s.
    p: 2, q: 2, MSE: 215.53414485111696, parameters: [ 1.57794589 -0.58052086  0.11120508 -1.19728612  0.10304867]
    
    Done with shop 260 after 26.602319599999998s.
    p: 1, q: 2, MSE: 373.56839408712653, parameters: [ 0.68711732 26.18318107 -0.44103851  0.04996526]
    
    Done with shop 315 after 23.080452199999996s.
    p: 2, q: 2, MSE: 323.6651118211104, parameters: [ 0.15121227  0.8587056  -0.59962977  0.24356502 -0.74502043]
    
    Done with shop 375 after 28.21356s.
    p: 1, q: 3, MSE: 229.71010897085569, parameters: [ 0.94607106  2.50898613 -0.7200216  -0.07855872 -0.38468647]
    
    Done with shop 548 after 30.800740900000008s.
    p: 3, q: 3, MSE: 335.3657714992873, parameters: [-0.00517792  0.15808891  0.83531331  0.66043892  0.03498125 -0.25753891
     -1.09691392]
     
    Done with shop 560 after 27.316482400000012s.
    p: 3, q: 3, MSE: 1076.1151601533625, parameters: [ 0.85036988  0.42865143 -0.27594663 -0.62356542 -0.51650497 -0.8367294
      0.183589  ]
      
    Done with shop 750 after 25.378438700000004s.
    p: 3, q: 3, MSE: 567.1980406143393, parameters: [ 1.13717765 -0.41880113  0.27862219  0.61953468 -1.02526741  0.39340039
     -0.22829494]
     
    Done with shop 897 after 27.76999459999999s.
    p: 3, q: 3, MSE: 12173.37622525072, parameters: [ 0.76842068 -0.58223577  0.79557697 -0.32767416 -0.53719915  0.08873952
     -0.7993756 ]
     
    Done with shop 1332 after 28.491905599999996s.
    p: 2, q: 3, MSE: 3299.616715119028, parameters: [ 1.2264627  -0.23596949  0.61406472 -0.80299952 -0.40896393  0.06080798]
    
    Done with shop 1750 after 23.425931100000014s.
    p: 2, q: 2, MSE: 429.40212750275066, parameters: [ 1.50992107 -0.50376428 -0.32331789 -1.06310663 -0.06116204]
    
    Done with all time series after 269.5872852s.
    """


if __name__ == '__main__':
    main()
