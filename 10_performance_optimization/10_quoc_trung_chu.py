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


def run_default(ds1, ds2):
    """ Default implementation using nested loops.

    :param ds1: list with points
    :param ds2: list with points
    :return: list of distances
    """
    result = []
    for p1 in ds1:
        for p2 in ds2:
            result.append(euclidian_distance(p1, p2))

    return result


def run_parallel(ds1, ds2):
    """ Run the calculation using multiprocessing.

    :param ds1: list with points
    :param ds2: list with points
    :return: list of distances
    """
    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.starmap(euclidian_distance, [(p1, p2) for p1 in ds1 for p2 in ds2])
    pool.close()

    return result


def run_vectorized(ds1, ds2):
    """ Run the calculation using vectorization

    :param ds1: numpy array with points
    :param ds2: numpy array with points
    :return: distance matrix (2000x2000)
    """
    # create new axis in ds1 to subtract all points of ds2 from all points of ds1
    return np.sqrt((np.square(ds1[:, np.newaxis] - ds2).sum(axis=2)))


def compare_results():
    """ Check if all methods return the same result by comparing the sum of all distances.

    :return: boolean
    """
    ds1 = np.random.rand(100, 2) * 100
    ds2 = np.random.rand(100, 2) * 100

    result_d = np.sum(run_default(ds1.tolist(), ds2.tolist()))
    result_p = np.sum(run_parallel(ds1.tolist(), ds2.tolist()))
    result_c = np.sum(euclid_c(ds1.tolist(), ds2.tolist()))
    result_v = np.sum(np.sum(run_vectorized(ds1, ds2)))

    return result_d == result_p == result_c == result_v


def main():
    setup = "from __main__ import euclidian_distance, run_default, run_parallel, run_vectorized;" \
            "from euclid import euclid_c;" \
            "import numpy as np; " \
            "ds1 = np.random.rand(2000, 2) * 100; " \
            "ds2 = np.random.rand(2000, 2) * 100"

    # number of benchmarking iterations
    number = 10
    print("Default: {}s"
          .format(timeit.timeit("run_default(ds1.tolist(), ds2.tolist())", setup=setup, number=number) / number))
    print("Parallel: {}s"
          .format(timeit.timeit("run_parallel(ds1.tolist(), ds2.tolist())", setup=setup, number=number) / number))
    print("Cython: {}s"
          .format(timeit.timeit("euclid_c(ds1.tolist(), ds2.tolist())", setup=setup, number=number) / number))
    print("Vectorization: {}s"
          .format(timeit.timeit("run_vectorized(ds1, ds2)", setup=setup, number=number) / number))

    print("All results equal: {}".format(compare_results()))

    """ Results: (with 10 iterations, CPU has 12 threads)
    Default: 2.49436669s
    Parallel: 2.0338920399999996s
    Cython: 1.7544557499999995s
    Vectorization: 0.11555078999999964s
    """


if __name__ == '__main__':
    main()
