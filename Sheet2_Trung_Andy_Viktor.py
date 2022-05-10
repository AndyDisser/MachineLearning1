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


def k_means(dataset, ncluster):
    """
    K-means algorithm for clustering dataset.

    Inputs:
    - dataset : the given dataset
    - ncluster : the number of cluster that we want to have

    Outputs:
    - an array containing the index of the cluster to which each element belongs
    - an array containing the coordinates of the final chosen centroids
    ...
    """

    # Get the number of rows.
    n = dataset.shape[0]

    # Initialize ncluster centroid first using k-means++
    centroid_arr = k_means_plus(dataset, ncluster)
    
    # Initialize the controids arr that will be updated after each main iteration.
    updated_centroid_arr = np.zeros((ncluster, 4))

    k=1

    # Loop until the old and new centroid array are the same
    while True:

        # Add new axis  (Broadcasting)
        new_dataset = dataset[np.newaxis,:,:]

        # Add new axis  (Broadcasting)
        new_centroid_arr = centroid_arr[:,np.newaxis,:]

        # Distance matrix (contains the distance of each point to each centroids)
        dist_matrix = new_dataset-new_centroid_arr

        # Compute the euclidean distance, subtract each centroid point from the
        # dataset, square each element and take the sum and lastly the root.
        # (broadcasting):  (1 x 150 x 4) . (3 x 1 x 4) = (3 x 150 x 4)
        dist_matrix = np.sqrt(np.apply_over_axes(np.sum, np.power(dist_matrix, 2), 2))

        # Take the nearest cluster for each element.
        cluster_indices_arr = np.apply_over_axes(np.argmin, dist_matrix[:, :, 0], 0)[0]

        i = 0
        # Filter
        while i < ncluster:
            # Take all the point that belongs to cluster i.
            filter_arr = cluster_indices_arr == i

            # Compute their mean.
            new_centroid = np.average(dataset[filter_arr], axis=0)

            # Set the mean as the new controid
            updated_centroid_arr[i,:] = new_centroid

            i += 1

        # If the centroid array does not change after update.
        if np.all(centroid_arr == updated_centroid_arr):
            break  # Stop the iteration.

        # Otherwise set the current centroid arr to the updated centroid arr.
        centroid_arr = copy.deepcopy(updated_centroid_arr)

        k += 1

    return cluster_indices_arr, centroid_arr


def k_means_plus(dataset, ncluster):
    """
    Initialize ncluster centroids.

    Inputs:
    dataset : given dataset
    ncluster : number of clusters

    Output : an array containing centroid coordinates. 
    """

    # Get the number of rows.
    n = dataset.shape[0]
    # Generate a random integer between 0 and n-1 (in case for iris data: 149)
    rand_index = np.random.randint(n)

    # Initialize a distant matrix with rows are the datapoint and columns are the centroid.
    dist_matrix = np.full((n, ncluster), float('inf'))

    # Initialize cluster indices array to be returned.
    cluster_indices = np.zeros(ncluster)

    centroid_arr = np.zeros((ncluster, 4))

    k = 0  # Run index
    curr_index = rand_index  # Initalize current index
    while k < ncluster:

        cluster_indices[k] = curr_index  # Add the index to the cluster_indices
        curr_centroid = iris_data[curr_index]  # Current centroid

        centroid_arr[k, :] = curr_centroid

        # Computing the square eucli. distance of all points to this centroid:
        # Subtract curr_elem from the dataset (broadcasting), then square each
        # element in the resulting matrix, then compute the sum of each row. 
        dist_from_centroid = np.apply_over_axes(np.sum, np.power(dataset-curr_centroid, 2), 1)

        # Take the root to receive eucl. distance
        dist_from_centroid = np.sqrt(dist_from_centroid)

        # Then store it vertically (reshape) in the distant matrix.
        dist_matrix[:, k] = dist_from_centroid.reshape(n)

        # We need to compute the distance of each point to its nearest chosen centroid.
        dist_of_nearest_centroid_arr = np.apply_over_axes(np.amin, dist_matrix, 1)

        # Then find the index of the point having the furthest distance from its nearest centroid.
        max_index = np.argmax(dist_of_nearest_centroid_arr)

        # Set this point to be the next centroid.
        curr_index = max_index
        k += 1

    # Return it as integer array
    return centroid_arr


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

        # Merge (compute the centroid)
        centroid = np.divide(data[i]+data[j], 2)

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

    # ============ Aufgabe 1.1: Test on Iris and plot 2D Scatter =====================
    cluster_indices_arr, centroid_arr = k_means(iris_data[:,:], 3)

    print("Array containing the indices of clusters to which the points belong:")
    print(cluster_indices_arr)
    print("Array containing the coordinates of the final centroids:")
    print(centroid_arr)

    # Filter array for the first cluster
    filter_arr_cluster_0 = cluster_indices_arr == 0
    # Filter array for the second cluster
    filter_arr_cluster_1 = cluster_indices_arr == 1
    # Filter array for the third cluster
    filter_arr_cluster_2 = cluster_indices_arr == 2

    # Filter the points that belong to a their certain cluster.
    cluster0 = iris_data[filter_arr_cluster_0]
    cluster1 = iris_data[filter_arr_cluster_1]
    cluster2 = iris_data[filter_arr_cluster_2]

    f1, axarr = plt.subplots(1,2, figsize=(10,5))        # Define the plot axes.
    
    # Plot the 3 different cluster with 3 different color for the first plot.
    axarr[0].scatter(cluster0[:, 0], cluster0[:, 1], s=10, color='blue')
    axarr[0].scatter(cluster1[:, 0], cluster1[:, 1], s=10, color='purple')
    axarr[0].scatter(cluster2[:, 0], cluster2[:, 1], s=10, color='lightgreen')      

    # Plot the three centroid with color red for the first plot.
    axarr[0].scatter(centroid_arr[0, 0], centroid_arr[0, 1], s=25, color='red')
    axarr[0].scatter(centroid_arr[1, 0], centroid_arr[1, 1], s=25, color='red')
    axarr[0].scatter(centroid_arr[2, 0], centroid_arr[2, 1], s=25, color='red')

    # Plot the 3 different cluster with 3 different color for the second plot.
    axarr[1].scatter(cluster0[:, 2], cluster0[:, 3], s=10, color='blue')
    axarr[1].scatter(cluster1[:, 2], cluster1[:, 3], s=10, color='purple')
    axarr[1].scatter(cluster2[:, 2], cluster2[:, 3], s=10, color='lightgreen')   

    # Plot the three centroid with color red for the second plot.
    axarr[1].scatter(centroid_arr[0, 2], centroid_arr[0, 3], s=25, color='red')
    axarr[1].scatter(centroid_arr[1, 2], centroid_arr[1, 3], s=25, color='red')
    axarr[1].scatter(centroid_arr[2, 2], centroid_arr[2, 3], s=25, color='red')
    
    # Set the title and the axes names.
    axarr[0].set_xlabel("Sepal length")
    axarr[0].set_ylabel("Sepal width")
    axarr[0].set_title("Sepal")

    axarr[1].set_xlabel("Petal length")
    axarr[1].set_ylabel("Petal width")
    axarr[1].set_title("Petal")


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

    linkage_matrix = agglomerative_clustering(iris_data[:10,:])  # Compute the linkage matrix.

    f2, myplot2 = plt.subplots(figsize=(10,5)) 

    myplot2.set_title("Hierarchical Clustering Dendrogram")
    myplot2.set_xlabel("Index of the data in the dataset (start from 0)")
    myplot2.set_ylabel("Distance between clusters")
    dendrogram(linkage_matrix)
    plt.show()





