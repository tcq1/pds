import pandas as pd
import numpy as np
from scipy.optimize import minimize
from timeit import default_timer as timer


# dataframe from given cost file
df = pd.read_csv('cost.csv', header=None)


def grid_search():
    """ Implementation of grid search to find global optimum

    :return: [function_value x y]
    """

    # choose [0, 0] as starting point
    best = np.array([cost_function(np.array([0, 0])),
                     0, 0])

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if cost_function(np.array([i, j])) < best[0]:
                best = np.array([cost_function(np.array([i, j])),
                                 i, j])

    print('Grid search: optimum:\n{}'.format(best))

    return best


def cost_function(params):
    """ Returns the cost by looking up in the cost dataframe with the parameters as row/column

    :param params: numpy array
    :return: int
    """
    # since we can only use positive integers to use iloc: cast possible float values to int
    x, y = params.astype(int)
    if x < 0:
        x = 0
    if y < 0:
        y = 0
        
    return df.iloc[x][y]


def nelder_mead(dimensions, alp, gam, rho, sig, thr):
    """ Implementation of the Nelder Mead algorithm. Returns a list of the optimal parameter values

    :param dimensions: number of dimensions
    :param alp: parameter for calculation of reflection
    :param gam: parameter for calculation of expansion
    :param rho: parameter for calculation of contraction
    :param sig: parameter for calculation of shrink
    :param thr: threshold value
    :return: [function_value param_opt_1 param_opt_2 ...]
    """

    # initialize starting points
    points = generate_random_point(dimensions, 0, min(df.shape))
    for i in range(dimensions):
        points = np.vstack((points, generate_random_point(dimensions, 0, min(df.shape))))

    iteration = 0
    last_iteration = None

    while np.std(points[:, 0]) > thr and not np.array_equal(last_iteration, points):
        # sort points and calc centroid
        points = points[points[:, 0].argsort()]
        centroid = calc_centroid(points)

        # store to compare in the next iteration if anything changed
        last_iteration = np.copy(points)
        print('Iteration {}: \nPoints: \n{}, \nCentroid: \n{}'.format(iteration, points, centroid))

        # decide replacement method
        if points[0][0] <= reflection(alp, points[0], centroid)[0] < points[-2][0]:
            xr = reflection(alp, points[-1], centroid)
            xe = expansion(gam, xr, centroid)
            if xe[0] < xr[0]:
                print('Performing reflection')
                points[-1] = xe
            else:
                print('Performing expansion')
                points[-1] = xr
        else:
            xc = contraction(rho, points[-1], centroid)
            if xc[0] < points[-1][0]:
                print('Performing contraction')
                points[-1] = xc
            else:
                print('Performing shrink')
                points = shrink(sig, points)

        iteration += 1
        print('std(points): {}\n'.format(np.std(points[:, 0])))

    # restart algorithm if stuck
    if np.array_equal(last_iteration, points):
        print('\n----- Simplex is stuck... retry again -----\n')
        return nelder_mead(dimensions, alp, gam, rho, sig, thr)

    # sort points again before returning
    points = points[points[:, 0].argsort()]

    return points


def calc_centroid(points):
    """ Calculates centroid of given points

    :param points: [function_value param1 param2 ...]
    :return: point
    """

    return update_point(np.mean(np.delete(points, -1, 0), axis=0))


def reflection(alp, worst_point, centroid):
    """ Calculates reflection of the worst point

    :param alp: reflection coefficient
    :param worst_point: point that has to be reflected
    :param centroid: centroid
    :return: reflected point
    """
    return update_point(centroid + alp * (centroid - worst_point))


def expansion(gam, xr, centroid):
    """ Calculates expanded reflection of the worst point

    :param gam: expansion coefficient
    :param xr: reflected point
    :param centroid: centroid
    :return: expanded reflected point
    """
    return update_point(centroid + gam * (xr - centroid))


def contraction(rho, worst_point, centroid):
    """ Calculates contraction of the worst point

    :param rho: contraction coefficient
    :param worst_point: point that has to be contracted
    :param centroid: centroid
    :return: contracted point
    """
    return update_point(centroid + rho * (worst_point - centroid))


def shrink(sig, points):
    """ Calculates all shrunk points

    :param sig: shrinking coefficient
    :param points: all points
    :return: new list of points
    """
    for i in range(1, len(points)):
        points[i] = update_point(points[0] + sig * (points[i] - points[0]))

    return points


def update_point(point):
    """ Usually data points are stored as [function_value param1 param2 ...]
    This function updates the function value since after some calculations function_value is invalid
    Also sets negative values to 0

    :param point: [function_value_invalid param1 param2 ...]
    :return: [function_value_valid param1 param2 ...]
    """

    # make sure points can't get below 0 and not above cost matrix
    for i in range(1, len(point)):
        if point[i] < 0:
            point[i] = 0
        if point[i] > df.shape[i-1]:
            point[i] = df.shape[i-1] - 1

    point = np.append(cost_function(point[1:, ]), point[1:, ])

    return point


def generate_random_point(dimensions, low, high):
    """ Creates a random starting point for the nelder mead algorithm, very first value of a row is the function value

    :param dimensions: number of parameters
    :param low: lowest possible value
    :param high: highest possible value
    :return: numpy array of form [function_value param1 param2 ...]
    """

    params = np.random.randint(low=low, high=high, size=dimensions)
    return np.append(cost_function(params), params)


def random_restart(dimensions, alp, gam, rho, sig, thr, iterations):
    """ Calls the nelder mead algorithm multiple times to find the best optimum out of several optima

    :param dimensions: number of dimensions
    :param alp: parameter for calculation of reflection
    :param gam: parameter for calculation of expansion
    :param rho: parameter for calculation of contraction
    :param sig: parameter for calculation of shrink
    :param thr: threshold value
    :param iterations: number of nelder mead iterations
    :return: [function_value_opt x_opt y_opt]
    """
    points = []
    for i in range(iterations):
        print('\n{}. Iteration of Nelder Mead:'.format(i+1))
        points.append(nelder_mead(dimensions, alp, gam, rho, sig, thr))

    best = sorted(points, key=lambda x: x[0][0])[0]
    print('Best optimum was \n{}'.format(best))

    return best


def function_task_2(params):
    """ Function given by task

    :param params: [x, y]
    :return: function value
    """
    x, y = params

    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2


def gradient_task_2(params):
    """ Calculate gradient value of function

    :param params: [x, y]
    :return: gradient
    """
    x, y = params
    x_grad = 2 * (x*y**6 + x*y**4 - 2*x*y**3 - x*y**2 - 2*x*y + 2.625*y**3 + 2.25*y**2 + 1.5*y + 3*x - 6.375)
    y_grad = 2 * (3*x**2*y**3 + 2*x**2*y**2 + x**2*y - 6*x**2 + 13.875*x)

    return [x_grad, y_grad]


def task2():
    """ Calculate optimal parameters of the given function by using scipy.optimize.minimize and compare runtime
    with/without gradient
    """
    x0 = np.array([1, 1])

    print('----- TASK 2 -----')
    print('without pre-calculated gradient:')
    start1 = timer()
    result = minimize(function_task_2, x0=x0, method='CG')
    end1 = timer()
    print('Time elapsed: {}'.format(end1 - start1))
    print(result)

    print('with pre-calculated gradient:')
    start2 = timer()
    result = minimize(function_task_2, x0=x0, method='CG', jac=gradient_task_2)
    end2 = timer()
    print('Time elapsed: {}'.format(end2 - start2))
    print(result)


def main():
    print('----------------- Nelder Mead -----------------')
    random_restart(2, 1, 2, 0.5, 0.5, 2, 100)

    # grid_search()
    # task2()


if __name__ == '__main__':
    main()
