import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from scipy.spatial import distance_matrix


def euclidian_distance(point_a, point_b):
    """ Calculates euclidian distance between both points

    :param point_a: numpy array
    :param point_b: numpy array
    """
    return np.linalg.norm(point_a - point_b)


def single_link_distance(cluster_a, cluster_b):
    """ Calculates single link distance between two clusters.

    :param cluster_a: numpy array of shape [n_points, n_dimensions]
    :param cluster_b: numpy array of shape [n_points, n_dimensions]
    """

    return min(euclidian_distance(point_a, point_b) for point_a in cluster_a for point_b in cluster_b)


def complete_link_distance(cluster_a, cluster_b):
    """ Calculates complete link distance between two clusters.

    :param cluster_a: numpy array of shape [n_points, n_dimensions]
    :param cluster_b: numpy array of shape [n_points, n_dimensions]
    """
    return max(euclidian_distance(point_a, point_b) for point_a in cluster_a for point_b in cluster_b)


def average_link_distance(cluster_a, cluster_b):
    """ Calculates average link distance between two clusters.

    :param cluster_a: numpy array of shape [n_points, n_dimensions]
    :param cluster_b: numpy array of shape [n_points, n_dimensions]
    """
    dist_sum = 0
    for point_a in cluster_a:
        for point_b in cluster_b:
            dist_sum += euclidian_distance(point_a, point_b)

    return sum(euclidian_distance(point_a, point_b)
               for point_a in cluster_a for point_b in cluster_b) / (len(cluster_a) * len(cluster_b))


def centroid_link_distance(cluster_a, cluster_b):
    """ Calculates centroid link distance between two clusters.

    :param cluster_a: numpy array of shape [n_points, n_dimensions]
    :param cluster_b: numpy array of shape [n_points, n_dimensions]
    """
    centroid_a = np.mean(cluster_a, axis=0)
    centroid_b = np.mean(cluster_b, axis=0)

    return euclidian_distance(centroid_a, centroid_b)


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
        centroid = np.random.random(2) * 15
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
# def distance_matrix(points, distance_function):
#     dist_mat = np.zeros((len(points), len(points)), dtype=np.float)
#     for i in range(len(points)):
#         for j in range(len(points)):
#             dist_mat[i][j] = distance_function(points[i], points[j])
#
#     return dist_mat


def find_closest_clusters(clusters, distance_function):
    """ Finds the two clusters that have the smallest distance based on the given distance_function.

    :param clusters: List of clusters
    :param distance_function: one of the functions that are defined on the top of this python file
    :return: indices of the two clusters that have to be merged
    """
    # get first two elements as pivot
    to_merge = [0, 1]
    for i in range(len(clusters)):
        for j in range(len(clusters)):
            # make sure to not compare the same cluster to itself
            if i == j:
                continue
            if distance_function(np.array(clusters[i]), np.array(clusters[j])) < \
                    distance_function(np.array(clusters[to_merge[0]]), np.array(clusters[to_merge[1]])):
                to_merge = [i, j]

    return to_merge


def append_cluster(cluster_a, cluster_b):
    """ Appends cluster_b to cluster_a

    :param cluster_a: array of shape [n_points, n_dimensions]
    :param cluster_b: array of shape [n_points, n_dimensions]
    """
    for point in cluster_b:
        cluster_a.append(point)

    return cluster_a


def assign_points_to_clusters(clusters):
    """ Assigns all points to clusters like in the assignment method of the kMeans algorithm.
    Simplifies handling of output of both methods

    :param clusters: output of agnes()
    :return: dictionary with index of cluster center (label) as key and numpy array of assigned points as value
    """
    # initialize assignment
    assignment = {i: None for i in range(len(clusters))}

    # iterate over clusters
    for i in range(len(clusters)):
        for point in clusters[i]:
            if assignment[i] is None:
                assignment[i] = np.atleast_2d(np.array(point))
            else:
                # else stack point to existing numpy array
                assignment[i] = np.vstack([assignment[i], point])

    return assignment


def agnes(points, distance_function=single_link_distance, stop_distance=None):
    """ Implementation of the agnes clustering algorithm.

    :param points: dataset of points
    :param distance_function: one of the functions that are defined on the top of this python file
    :param stop_distance: stops the tree building process if stop_distance is exceeded (simulates the tree cut)
    :return: clusters
    """
    # convert to list TODO: try to make work with np arrays
    clusters = points.tolist()
    for i in range(len(clusters)):
        clusters[i] = [clusters[i]]
    while len(clusters) > 1:
        start = timer()
        to_merge = find_closest_clusters(clusters, distance_function)
        end = timer()
        print('Finding this merge took {}s'.format(end - start))
        print('Current shortest distance: {}'.format(distance_function(np.array(clusters[to_merge[0]]),
                                                                       np.array(clusters[to_merge[1]]))))
        if stop_distance is not None:
            if distance_function(np.array(clusters[to_merge[0]]), np.array(clusters[to_merge[1]])) > stop_distance:
                break
        append_cluster(clusters[to_merge[0]], clusters.pop(to_merge[1]))

    return clusters


def agnes2(points):
    clusters = []

    dist = distance_matrix(points, points)
    dist[dist == 0] = np.inf

    for i in range(len(points)):
        clusters[i] = [points[i]]

    while len(clusters) > 1:
        pass

    print(dist)
    print(clusters)

    return True


def test(data):
    combined_clusters = {}
    count = 0

    while len(data) > 1:
        dist = distance_matrix(data, data)
        dist[dist == 0] = np.inf

        min_dist_index = np.unravel_index(dist.argmin(), dist.shape)
        min_dist = dist[min_dist_index]

        closest_clusters = data.index[list(min_dist_index)]

        cluster_name = 'c{}'.format(count)
        combined_clusters[cluster_name] = list(closest_clusters) + [min_dist]

        data.loc[cluster_name] = data.loc[closest_clusters].mean()
        data = data.drop(closest_clusters)

        count += 1

    return combined_clusters

# hierarchical clustering
# -------------------------------------------------------------------------------------------------


def plot_clustering(assignment):
    """ Plots data points with assigned labels.

    :param assignment: Dictionary with assignment of points to a cluster
    """
    for cluster in assignment.values():
        # generate random color for this cluster
        color = np.random.rand(3, )
        for point in cluster:
            plt.scatter(point[0], point[1], color=color)

    plt.show()


def main():
    # get data
    file_path = 'data-clust.csv'
    df = pd.read_csv(file_path)
    dataset = df.to_numpy()

    start_clustering = timer()

    # kMeans
    # centroids = k_means(dataset, k=4)
    # assignment = assign_points_to_centroid(dataset, centroids)

    # clusters = agnes(dataset, stop_distance=0.1)
    # assignment = assign_points_to_clusters(clusters)
    print(agnes2(dataset))

    end_clustering = timer()
    print('Clustering took {}s'.format(end_clustering - start_clustering))

    # plotting
    # plot_clustering(assignment)


if __name__ == '__main__':
    main()
