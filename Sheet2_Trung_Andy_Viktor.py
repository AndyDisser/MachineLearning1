"""
Name - Matrikelnummer 
1) Pham, Ngoc Anh Trung - 7176267
2) Viktor Vironski - 4330455
3) Andy Disser - 5984875

Exercise Sheet 2
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram


def get_points_of_cluster(linkage_matrix, row_index):
    """
    Recursive function which returns an array consisting of indices of original 
    points in the cluster with the given row index in the given linkage matrix.
    
    Inputs:
    - linkage_matrix : the linkage matrix
    - row_index : the row index of the cluster in linkage matrix

    Output:
    - array containing the indices of original points lying in this cluster
    """

    # Base case (If it was a cluster composed of two original points)
    if linkage_matrix[row_index, 3] == 2:
        return np.array([linkage_matrix[row_index, 0], linkage_matrix[row_index, 1]])

    else:  # General case
        
        n = linkage_matrix.shape[0] + 1   # The number of original points
        i = linkage_matrix[row_index, 0]  # The left subcluster
        j = linkage_matrix[row_index, 1]  # The right subcluster

        left_arr = np.array([linkage_matrix[row_index, 0]])
        right_arr = np.array([linkage_matrix[row_index, 1]])

        # If the subcluster i or j is not trivial (not a single-element-cluster)
        if (i >= n):
            left_arr = get_points_of_cluster(linkage_matrix, (i-n).astype(int))
        if (j >= n):
            right_arr = get_points_of_cluster(linkage_matrix, (j-n).astype(int))

        return np.concatenate([left_arr, right_arr])


def agglomerative_clustering(data):
    """
    Cluster the data using the agglomerative method.
    Distance is defined based on the distance between centroid.

    Inputs:
    data : the given dataset

    Outputs:
    - the linkage matrix
    """

    n = data.shape[0]
    k = 0
    
    # According to the format of the documentation. 
    linkage_matrix = np.zeros((n-1, 4))

    # A copied of the main data to work with.
    worked_data = copy.deepcopy(data)
    
    # Array counting the original observations in each newly formed cluster
    # Newly form cluster are indexed from n to 2n-2
    # Initially each cluster has a number of original observation of 1.
    count_arr = np.ones(2*n-1)

    # Create a matrix to keep track of the index of points.
    indices_matrix = np.zeros((n,n,2))
    vec1 = np.arange(n)
    vec2 = np.arange(n)
    indices_matrix[:,:,0] = vec1[:, np.newaxis]
    indices_matrix[:,:,1] = vec2

    # There are in total n-1 iterations
    while k < n-1:
        
        # Create 2 broadcasted copies of the dataset in order to compute the distance (linkage) matrix.
        data1 = worked_data[np.newaxis, :, :]
        data2 = worked_data[:, np.newaxis, :]

        # Compute the distance/proximity matrix.
        dist_matrix = np.sqrt(np.apply_over_axes(np.sum, np.power(data1-data2, 2), 2))

        # Replace the zeros with infinity for the sake of computing the minimum.
        dist_matrix[dist_matrix == 0] = np.inf

        # Get the minimum distance in the distance
        minval = np.nanmin(dist_matrix)
        
        # Get the index pair (i,j) where the distance minimal
        i, j = np.argwhere(dist_matrix == minval)[0][:2]
        
        np.array([indices_matrix[i, j, 0], indices_matrix[i, j, 1]])

        # The true indicies of the point which we have been kept track of through indices matrix.
        mapped_i = indices_matrix[i, j, 0]
        mapped_j = indices_matrix[i, j, 1]

        # Store them in the linkage matrix.
        linkage_matrix[k, 0] = mapped_i
        linkage_matrix[k, 1] = mapped_j
        # Store the min distance.
        linkage_matrix[k, 2] = minval
        # The number of original observations in the newly form cluster 
        # is equal to the sum of the number of original observations which were already
        # lie in the subcluster.
        count_arr[n+k] = count_arr[mapped_i.astype(int)] + count_arr[mapped_j.astype(int)]
        linkage_matrix[k, 3] = count_arr[n+k]

        # Return the indices of original points
        indices = get_points_of_cluster(linkage_matrix, k).astype(int)

        # Get the points with these indices in the dataset (data[indices]) and 
        # compute the centroid (merge)
        centroid = np.divide(np.apply_over_axes(np.sum, data[indices], 0), count_arr[n+k])[0]

        # Replace the row with smaller index with the merged row.
        worked_data[min(i,j), :] = centroid
        # Replace the other row with the larger index with only infinity, making it irrelevant.
        worked_data[max(i,j), :] = np.ones(1)*np.inf

        # Update the indicies matrix.
        indices_matrix[min(i, j), :, 0] = n+k  # Update row
        indices_matrix[:, min(i, j), 1] = n+k  # Update column

        k += 1

    return linkage_matrix

if __name__ == '__main__':

    # Import and access the data
    iris = datasets.load_iris()
    iris_data = iris['data']

    # ============== Aufgabe 1.2 ================================

    # Initialize an array containing mean distance for each ncluster-value
    mean_distances_arr = np.zeros(10) 

    # Run k-means for all ncluster-value in [1:10]
    for ncluster in range(1,11):
        
        cluster_indices_arr, centroid_arr = k_means(iris_data[:,:], ncluster)

        total_within_cluster_dist = 0  # Sum of all distances of each point to their respective nearest centroid.

        for k in range(ncluster):  # For each cluster do..

            # Filter array for the cluster k
            filter_arr_cluster_k = cluster_indices_arr == k

            # Filter the points that belong to their certain cluster.
            clusterk = iris_data[filter_arr_cluster_k]

            # Compute the distance of elements in this cluster to its according centroid
            dist_to_nearest_centroid_arr = np.sqrt(np.apply_over_axes(np.sum, np.power((clusterk-centroid_arr[k]),2), 1))

            # Sum it up and add to the culumative sum
            total_within_cluster_dist += np.sum(dist_to_nearest_centroid_arr)

        # Compute the mean distance over all data point.
        mean_distance = np.divide(total_within_cluster_dist, iris_data.shape[0])

        # Store it in an array to plot.
        mean_distances_arr[ncluster-1] = mean_distance

    # Start plotting here.
    x = np.linspace(1, 10, num=10)  # Create the array: [1,2,...,10] for the x-axis

    f2, myplot = plt.subplots(figsize=(10,10))  # Define the plot.

    myplot.set_title('Elbow')                                       # Set title
    myplot.set_xlabel('Number of clusters')                         # Set x axis title
    myplot.set_ylabel('Mean distance to nearest centroid')          # Set y axis title
    
    plt.xticks(np.arange(min(x), max(x)+1, 1))                      # Set tick for x axis
    myplot.plot(x, mean_distances_arr)                              # Plot the line


    # ================== Aufgabe 2 ===========================

    linkage_matrix = agglomerative_clustering(iris_data[:15,:])  # Compute the linkage matrix.

    f2, myplot2 = plt.subplots(figsize=(10,5)) 

    myplot2.set_title("Hierarchical Clustering Dendrogram")
    myplot2.set_xlabel("Index of the data in the dataset (start from 0)")
    myplot2.set_ylabel("Distance between clusters")
    dendrogram(linkage_matrix)
    plt.show()
