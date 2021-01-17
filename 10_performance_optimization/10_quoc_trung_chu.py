import multiprocessing as mp
import numpy as np
import math
import timeit

from euclid import euclid_c


def euclidian_distance(p1, p2):
    """ Calculate the euclidian distance between two points.

    :param p1: Point 1
    :param p2: Point 2
    :return: float
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def euclidian_distance_serial(ds1, ds2):
    """ Calculate the euclidian distance for every point in ds1 to every point in ds2

    This function only calculates but doesn't return anything since only performance is measured.

    :param ds1: numpy array with points
    :param ds2: numpy array with points
    :return: list of distances
    """
    result = []
    for p1 in ds1:
        for p2 in ds2:
            result.append(euclidian_distance(p1, p2))

    return result


def run_parallel(ds1, ds2):
    """ Run the calculation using multiprocessing.

    :param ds1: numpy array with points
    :param ds2: numpy array with points
    :return: list of result
    """
    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.starmap(euclidian_distance, [(p1, p2) for p1 in ds1 for p2 in ds2])

    return result


def run_vectorized(ds1, ds2):
    """ Run the calculation using vectorization

    :param ds1: numpy array with points
    :param ds2: numpy array with points
    :return: distance matrix (2000x2000)
    """
    # create new axis in ds1 to subtract all points of ds2 from all points of ds1
    return np.sqrt((np.square(ds1[:, np.newaxis] - ds2).sum(axis=2)))


def benchmark_function(function, ds1, ds2, display_string):
    """ Benchmark the functions.

    :param function: function to benchmark
    :param ds1: first dataset of points
    :param ds2: second dataset of points
    :param display_string: additional string identifier to display when starting the benchmark
    :return: sum of all distances for comparison
    """
    print("Starting {} calculation...".format(display_string))
    start = timeit.default_timer()
    result = np.sum(function(ds1, ds2))
    end = timeit.default_timer()
    print("Calculation took {}s.\n".format(end-start))

    return result


def main():
    # generate random points from 0 to 100
    ds1 = np.random.rand(2000, 2) * 100
    ds2 = np.random.rand(2000, 2) * 100

    result_s = benchmark_function(euclidian_distance_serial, ds1, ds2, "serial")   # ~7.6s
    result_p = benchmark_function(run_parallel, ds1, ds2, "parallel")              # ~22.8s
    result_c = benchmark_function(euclid_c, list(ds1), list(ds2), "cython")        # ~ 5.3s
    result_v = benchmark_function(run_vectorized, ds1, ds2, "vectorized")          # ~ 0.1s

    print("All calculations have the same solution: {}".format(result_s == result_p == result_c == result_v))


if __name__ == '__main__':
    main()
