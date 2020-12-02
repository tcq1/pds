import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer


def euclidian_distance(point_a, point_b):
    """ Calculates euclidian distance between both points

    :param point_a: numpy array
    :param point_b: numpy array
    """
    return np.linalg.norm(point_a - point_b)
# distance measures
# -------------------------------------------------------------------------------------------------
# kMeans


def initialize_centroids(k):
    """ Initialize random points with labels.

    :param k: number of points
    :return: np array, array
    """
    centroids = []
    # create k centroids
    for i in range(k):
        # random values from 0 to 15
        centroid = np.random.random(2)*15
        centroids.append(centroid)

    return np.array(centroids)


def get_closest_centroid(centroids, point):
    """ Finds the label of the closest centroid of a point

    :param centroids: np array of centroids
    :param point: point
    :return: index (or label)
    """
    # get closest centroid
    closest = min(centroids, key=lambda centroid: euclidian_distance(centroid, point))
    # return index of closest
    return np.where(np.all(centroids == closest, axis=1))[0][0]


def assign_points_to_centroid(points, clustering):
    """ Assigns all points to a cluster center.

    :param points: numpy array of points
    :param clustering: numpy array of cluster centers
    :return: dictionary with index of cluster center (label) as key and numpy array of assigned points as value
    """
    # initialize assignment
    assignment = {i: None for i in range(len(clustering))}
    # iterate over all points
    for point in points:
        # get index of closest centroid
        i = get_closest_centroid(clustering, point)
        if assignment[i] is None:
            # if no points assigned so far: assign point
            assignment[i] = np.atleast_2d(np.array(point))
        else:
            # else stack point to existing numpy array
            assignment[i] = np.vstack([assignment[i], point])

    return assignment


def calculate_new_centroids(clustering):
    """ Calculates new cluster centers.

    :param clustering: Dictionary with assignment of points to a cluster
    :return: centroids
    """
    centroids = []
    for label in clustering.keys():
        # if no points were assigned to cluster center then drop the cluster
        if clustering[label] is not None:
            centroids.append(np.mean(clustering[label], axis=0))

    return centroids


def k_means(points, k):
    """ Implementation of the kMeans clustering algorithm.

    :param points: dataset of points
    :param k: number of cluster centers
    """
    # get initial cluster centers
    clustering_ = initialize_centroids(k)

    # empty clustering
    clustering = np.array([])

    iterations = 0

    # repeat until clustering doesn't change
    while not np.array_equal(clustering, clustering_):
        clustering = clustering_
        # update clustering
        clustering_ = calculate_new_centroids(assign_points_to_centroid(points, clustering))
        iterations += 1

    print('Final cluster centers: {}'.format(clustering))
    # during the process it is possible that there are centroid that got no assigned points
    print('Dropped {} clusters'.format(k - len(clustering)))
    print('Done after {} iterations'.format(iterations))

    return clustering

# kMeans
# -------------------------------------------------------------------------------------------------
# hierarchical clustering


# hierarchical clustering
# -------------------------------------------------------------------------------------------------

def plot_clustering(assignment):
    """ Plots data points with assigned labels.

    :param assignment: Dictionary with assignment of points to a cluster
    """
    for cluster in assignment.values():
        # generate random color for this cluster
        color = np.random.rand(3,)
        for point in cluster:
            plt.scatter(point[0], point[1], color=color)

    plt.show()


def main():
    # get data
    file_path = 'data-clust.csv'
    df = pd.read_csv(file_path)
    dataset = df.to_numpy()

    # kMeans
    start_clustering = timer()

    centroids = k_means(dataset, k=4)
    assignment = assign_points_to_centroid(dataset, centroids)

    end_clustering = timer()
    print('Clustering took {}s'.format(end_clustering - start_clustering))

    # plotting
    plot_clustering(assignment)


if __name__ == '__main__':
    main()
