import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import minimize
from timeit import default_timer as timer


lam1 = None
lam2 = None
lam3 = None


def get_random_start_params(min_val, max_val):
    """ Generates random start values for x and y
    """
    return np.array([np.random.randint(min_val, max_val), np.random.randint(min_val, max_val)])


def ols(function, x, y):
    """ OLS algorithm using the scipy.optimize.minimize function
    """
    result = minimize(function, x0=np.array(get_random_start_params(0, len(x))), args=(x, y))

    return result.x, result.fun


def sse(a, x, y):
    """ Calculates sum of squared errors of a function a
    """
    result = np.sum((y - x @ a)**2)
    # print('SSE = {}'.format(result))

    return result


def sse_ridge(a, x, y):
    """ Calculates the solution using the formula for Ridge Regression
    """
    result = sse(a, x, y) + lam1 * np.sum(a**2)
    # print('SSE Ridge: {}'.format(result))

    return result


def sse_lasso(a, x, y):
    """ Calculates the solution using the formula for Lasso Regression
    """
    result = sse(a, x, y) + lam1 * np.sum(abs(a))
    # print('SSE Lasso: {}'.format(result))

    return result


def sse_net(a, x, y):
    """ Calculates the solution using the formula for Elastic Net
    """
    result = sse(a, x, y) + lam2 * np.sum(a**2) + lam3 * (np.sum(abs(a)))
    # print('SSE Net: {}'.format(result))
    return result


def matrix_solution(x, y):
    """ Calculates the solution using matrix calculations
    """
    return np.linalg.inv(x.transpose() @ x) @ x.transpose() @ y


def main():
    # prepare data
    file_path = 'data-OLS.csv'
    x, y = np.loadtxt(file_path, delimiter=',', dtype=float, skiprows=1, unpack=True)
    x = np.column_stack((x, np.ones(len(x))))

    # set lambda values
    global lam1, lam2, lam3
    lam1 = 10
    lam2 = 5
    lam3 = 5

    # perform calculations
    # benchmark ols
    ols_start = timer()
    m0, n0 = ols(sse, x, y)[0]
    ols_end = timer()
    # ridge regression
    m1, n1 = ols(sse_ridge, x, y)[0]
    # lasso regression
    m2, n2 = ols(sse_lasso, x, y)[0]
    # elastic net
    m3, n3 = ols(sse_net, x, y)[0]
    # benchmark matrix
    matrix_start = timer()
    m4, n4 = matrix_solution(x, y)
    matrix_end = timer()

    # print out times and functions
    print('Performance of OLS algorithm: {}'.format(ols_end-ols_start))
    print('Performance of matrix algorithm: {}'.format(matrix_end-matrix_start))

    solutions = {'sse': '{}x + {}'.format(m0, n0), 'ridge': '{}x + {}'.format(m1, n1),
                 'lasso': '{}x + {}'.format(m2, n2), 'net': '{}x + {}'.format(m3, n3),
                 'matrix': '{}x + {}'.format(m4, n4)}

    for key in solutions.keys():
        print('Function of {}: {}'.format(key, solutions[key]))

    # plot
    x = x[:, 0]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(x, y, s=8)
    ax.plot(x, m0 * x + n0, c='C0', label='sse')
    ax.plot(x, m1 * x + n1, c='C1', label='ridge')
    ax.plot(x, m2 * x + n2, c='C2', label='lasso')
    ax.plot(x, m3 * x + n3, c='C3', label='net')
    ax.plot(x, m4 * x + n4, c='C4', label='matrix')
    ax.legend(fontsize=12)
    plt.show()


if __name__ == '__main__':
    main()
