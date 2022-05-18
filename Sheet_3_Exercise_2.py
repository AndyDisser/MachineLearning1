"""
Name - Matrikelnummer 
1) Pham, Ngoc Anh Trung - 7176267
2) Viktor Vironski - 4330455
3) Andy Disser - 5984875

Exercise Sheet 2
"""

from random import random
from pyparsing import original_text_for
from sklearn import cluster, datasets
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
import maxflow as mf



def compute_mean_and_covariance(matrix):
    """
    (Auxiliary function for GMM)
    Compute mean and covariance matrix of a given matrix. Only
    design for 2D-array (i.e. matrix)

    Inputs
    - matrix : the given matrix

    Outputs:
    - the mean and the covariance matrix
    """

    # Let (m x n) be the dimension of the given matrix.
    n = matrix.shape[0]
    m = matrix.shape[1]

    # Compute mean -> (1 x n)
    mean = np.apply_over_axes(np.average, matrix, 0)

    # Tranpose mean -> (n x 1)
    t_mean = np.transpose(mean)

    # Transpose the matrix and make 2 copies from it -> (n x m)
    left = np.transpose(matrix)
    right = np.transpose(matrix)

    # (Broadcasted) Subtract the mean to their respective values
    left = left - t_mean
    right = right - t_mean

    # Add new axis to make the broadcast possible.
    left = left[:, np.newaxis, :]  # -> (m x 1 x n)
    right = right[np.newaxis, :, :]  # -> (1 x m x n)

    # Compute the covariance matrix using broadcasting.
    # 1. Multiply pairwise
    # 2. Sum it up
    # 3. Divide by n
    # 4. Adjust the dimension
    cov_matrix = np.apply_over_axes(np.average, np.multiply(left, right), 2).reshape(m, m)

    return mean, cov_matrix

def pdf(x, mean, cov_matrix):
    """
    (Auxiliary function for GMM)
    Compute the density of point x, given the mean and the covariance matrix

    Inputs:
    - x : the given point
    - mean : the given mean
    - cov_matrix : the given covariance matrix
    """

    m = x.shape[0]                          # Dimension of x
    a = x-mean                              # Compute x-mean
    inv = np.linalg.inv(cov_matrix)         # Compute the inverse of covariance matrix
    det = np.linalg.det(cov_matrix)         # Compute the determinant of covariance matrix

    # Compute the density at point x
    density = np.power(2*(np.pi), -m/2, dtype='float') * np.power(det, -0.5, dtype='float') \
         * np.exp(((-0.5) * np.matmul(np.matmul(a, inv), np.transpose(a))).reshape(1)[0])

    return density


def GMM(data, kcomps):
    """
    Cluster the dataset using Gaussian mixture model.
    
    Input:
    - data: the given dataset
    - kcomps: the number of cluster components we want to have

    Output:
    - means : an array containing the means
    - cov_matrices : an array containing the covariance matrices
    - weights : an array containing the weights
    - resp_matrix : responsibility matrix

    """

    # Let k = kcomps (notation)

    # Let (m x n) be the dimension of the given dataset.
    n = data.shape[0]
    m = data.shape[1]

    # Initialize mixture weights (equal)
    weights = np.full(kcomps, np.power(kcomps, -1, dtype='float'))

    # Initialize log likelihood function
    curr_log_likelihood = 0

    # Epsilon for stopping the iteration
    epsilon = 0.00000000001

    # =============== Spliting the dataset in k components ========================

    # Set up random indices array, make sure each component has at least m+1 elements. Otherwise
    # it could lead to the determinant of some covariance matrix equals 0

    ok = False  # Ok if each component has at least m+1 elements
    
    while not ok:
        ok = True
        random_indices = np.random.randint(0, kcomps, n)

        for i in range(kcomps):
            if np.count_nonzero(random_indices == i) < (m+1):  # If any component has less than m+1 elements
                ok = False
                continue  # repeat

        # print(random_indices)

    # Store all kcomps covariance matrices
    cov_matrices = np.zeros((kcomps, m, m))

    # Store all determinant of the covariance matrices
    det_cov_matrices = np.zeros(kcomps)

    # Store all kcomps means
    means = np.zeros((kcomps, m))

    # For each component in kcomps components
    for i in range(kcomps):
        subdata = data[random_indices == i]  # Get the data subset in ith split

        # Compute the mean and the covariance matrix of this split
        mean, cov_matrix = compute_mean_and_covariance(subdata)

        # Store the covariance matrix and its determinant
        cov_matrices[i], det_cov_matrices[i] = cov_matrix, np.linalg.det(cov_matrix)
        # Store the mean.
        means[i] = mean

    j = 1

    while True:
        # print(j, ". Iteration")
        # print("means:\n", means)
        # print("cov_matrices:\n", cov_matrices)
        # print("weights:\n", weights)
        # print("Log likelihood", curr_log_likelihood)

        # Compute the inverses of each covariance matrices.
        inv_cov_matrices = np.linalg.inv(cov_matrices)

        # Compute the determinant of each covariance matrices.
        det_cov_matrices = np.linalg.det(cov_matrices)

        j += 1
        # ================ Expectation step =============================================

        # Compute cumulatively the density matrix having the dimension of (n) x (kcomps) and representing
        # the density of a point in n points under a parameter-pair of kcomps pairs.
        dens_matrix = None
        
        # Broadcast -> (n x 1 x m)
        dens_matrix = data[:, np.newaxis, :]

        # Broadcast -> (1 x k x m)
        means = means[np.newaxis, :, :]

        # diff = x - mean (Storing this specific term for later use)
        # Resulting dimension: (n x 1 x m) - (1 x k x m) -> (n x k x m)
        diff = dens_matrix - means

        # Broadcast again -> (n x k x 1 x m), for the matrix multiplication
        dens_matrix = diff[:, :, np.newaxis, :]

        # Broadcast the inversed covariance matrices -> (1 x k x m x m)
        inv_cov_matrices = inv_cov_matrices[np.newaxis, :, :, :]

        # Broadcast -> (n x k x 1 x m) . (1 x k x m x m) = ( n x k x 1 x m),  
        # The last 2 values of the dimensions result from the matrix mult.
        dens_matrix = np.matmul(dens_matrix, inv_cov_matrices)

        # Broadcast the diff (x-mean) -> (n x k x m x 1), for the second matrix multiplication 
        diff = diff[:, :, :, np.newaxis]
        
        # Execute the second matrix multiplication -> (n x k x 1 x 1)
        dens_matrix = np.matmul(dens_matrix, diff)

        # Reshape -> (n x k x 1)
        dens_matrix = dens_matrix.reshape((n, kcomps, 1))
        #print(dens_matrix.shape)

        # Multiply with -0.5 then wrap the exponential function around each entry
        dens_matrix = np.exp(np.multiply(dens_matrix, -0.5))
        #print(dens_matrix)

        # This is the first term of the density formel containing the pi.
        firstterm = np.power(2*(np.pi), -m/2, dtype='float')

        # Raise each determinant to the power of -0.5
        det_cov_matrices = np.power(det_cov_matrices, -0.5, dtype='float')

        # Broadcast
        det_cov_matrices = det_cov_matrices[np.newaxis, :, np.newaxis]

        # Multiply it with the current result
        dens_matrix = np.multiply(det_cov_matrices, dens_matrix)

        # Then multiply it with the first term.
        dens_matrix = np.multiply(dens_matrix, firstterm)
        
        # Final reshaping to (n x k)
        dens_matrix = dens_matrix.reshape((n, kcomps))

        # Broadcast the weight array
        weights = weights[np.newaxis, :]

        # Weight each entry with the according values from weight array.
        weighted_entries = np.multiply(weights, dens_matrix)

        # Sum all the weighted densities.
        weighted_sum = np.apply_over_axes(np.sum, weighted_entries, 1)

        # Now compute the responsibility matrix from the weighted_entries and weighted_sum
        resp_matrix = np.divide(weighted_entries, weighted_sum)

        # Compute new log likelihood
        new_log_likelihood = np.sum(np.log(weighted_sum))

        # print("new log likelihood", new_log_likelihood)

        if (np.abs(curr_log_likelihood-new_log_likelihood) < epsilon):  # If overall model does not change much
            break       # Then stop
        
        curr_log_likelihood = new_log_likelihood  # Otherwise update log likelihood

        # ================ Maximization step =============================================

        # The matrix containing the total responsibilities of each k column
        total_resp_of_k = np.apply_over_axes(np.sum, resp_matrix, 0)

        # Transpose it
        total_resp_of_k = np.transpose(total_resp_of_k)

        # ======= Update means =====================
        # Transpose the responsibility matrix
        t_resp_matrix = np.transpose(resp_matrix)

        # Broadcast => (k x n x 1)
        t_resp_matrix = t_resp_matrix[:, :, np.newaxis]

        # Broadcast => (1 x n x m)
        temp_data = data[np.newaxis, :, :]

        # Multiply elementwise
        upd = np.multiply(t_resp_matrix, temp_data)

        # Then sum up and reshape to (k, m)
        upd = np.apply_over_axes(np.sum, upd, 1).reshape((kcomps, m))

        # Compute the new means array
        new_means = np.divide(upd, total_resp_of_k)

        # Update means
        means = new_means

        # ======= Update covariance matrices =====================

        # Add new axis -> (1 x n x m) 
        temp_data = data[np.newaxis, :, :]
        
        # Reshape means array
        temp_means = means.reshape((kcomps, m))

        # Add new axis -> ( kcomps x 1 x m)
        temp_means = temp_means[:, np.newaxis, :]

        # Compute the difference (x - mean) elementwise
        temp = temp_data - temp_means

        # Broadcast (k x n x m x 1) . (k x n x 1 x m) = (k x n x m x m)
        temp = np.matmul(temp[:, :, :, np.newaxis], temp[:, :, np.newaxis, :])

        # Broadcast the transposed responsibility matrix -> (kcomps x n x 1 x 1)
        t_resp_matrix = t_resp_matrix[:, :, :, np.newaxis]

        # Multiply
        temp = np.multiply(t_resp_matrix, temp)

        # Then compute the sum of the (mxm)-matrices
        temp = np.apply_over_axes(np.sum, temp, 1)

        # Reshape
        temp = temp.reshape((kcomps, m, m))
        # print(temp.shape)

        # Divide by the total resp. in each column for the according new covariance matrix
        new_cov_matrices = np.divide(temp, total_resp_of_k[:, :, np.newaxis])

        # ======= Update weights =====================
        
        # The matrix containing the total responsibilities of each k column
        total_resp_of_k = np.apply_over_axes(np.sum, resp_matrix, 0)

        # Compute new weights
        new_weights = np.divide(total_resp_of_k, n).reshape(3)

        # Update covariance matrices and weights
        cov_matrices = new_cov_matrices
        weights = new_weights

    return means.reshape(kcomps, m), cov_matrices, weights, resp_matrix



def make_ellipses(means, cov_matrices, ax):
    """
    Draw ellipses, taken from 
    https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    """

    colors = ["navy", "yellow", "darkorange"]
    for n, color in enumerate(colors):
        covariances = cov_matrices[n, :2, :2]

        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        
        ell = mpl.patches.Ellipse(
            means[n, :2], v[0], v[1], 180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

def hungarian_matching(cluster_index, gt_label):
    """
    Match the cluster to the actual classes with minimal cost.

    Inputs:
    - cluster_index : an array containing the cluster indices
    - gt_label : an array containing the actual classification of data

    Output:
    - the minimum number of missclassification / minimum cost
    """
    
    # The number of components / clusters
    kcomps = np.size(np.unique(gt_label))

    # Initialize the cost matrix
    cost_matrix = np.zeros((kcomps, kcomps))

    # Construct the cost matrix
    for i in range(kcomps):
        for j in range(kcomps):
            # Get all indices where the clustering result equals to i ( all indices of the
            # point which assigned to cluster i by the algorithm)
            indices = np.where(cluster_index == i)

            # The cost assigning cluster i to class j equals to the number of points that are assigned
            # to cluster i and at the same time not belong to class j.
            cost_matrix[i, j] = np.size(target[indices][target[indices] != j])

    origin_cost_matrix = cost_matrix

    # Step 1: subtract the minimum of the row from each entry of that row
    cost_matrix = cost_matrix - np.apply_over_axes(np.amin, cost_matrix, 1)

    # Step 2: subtract the minimum of the column from each entry of that column
    cost_matrix = np.transpose(np.transpose(cost_matrix) - np.apply_over_axes(np.amin, cost_matrix, 0))


    # Step 3: Draw minimum lines that cover all zeros.
    # For this step i will take reference from this page:
    # https://en.wikipedia.org/wiki/Hungarian_algorithm
    while True:
        
        # Initilization the needed variables.
        starred_matrix = np.zeros((kcomps, kcomps))
        primed_matrix = np.zeros((kcomps, kcomps))
        assigned_columns = set()
        assigned_rows = set()
        covered_columns = set()
        covered_rows = set()

        # Step 3.1 : Assign as much row as possible
        for i in range(kcomps):

            j = np.argmin(cost_matrix[i, :])    # The first entry in row i that equals 0

            if (j not in assigned_columns):
                assigned_rows.add(i)
                # Update starred matrix
                starred_matrix[i, j] = 1
            
            assigned_columns.add(j)             # Add it to the assigned columns

        while True:  

            # Step 3.2 : Cover all columns that having an assignment / having starred zero
            for i in range(kcomps):
                for j in range(kcomps):
                    if (starred_matrix[i, j] == 1):
                        covered_columns.add(j)

            # Step 3.3 : Find the zeros that are not lying under an assigned column and prime it
            indices = np.where(cost_matrix == 0)

            row_indices, column_indices = indices[0], indices[1]

            for i in range(np.size(indices[0])):
                # not lying under an assigned column
                if (column_indices[i] not in covered_columns): #starred_matrix[row_indices[i], column_indices[i]] == 0 :
                
                    primed_matrix[row_indices[i], column_indices[i]] = 1  # Prime this zero

                    
                    starred_zero_index = np.zeros(2)
                    there_is_starred_zero = False
                    for j in range(kcomps):
                        # Check if there is a starred zero on this line.
                        if starred_matrix [row_indices[i], j] == 1:
                            starred_zero_index = row_indices[i], j
                            there_is_starred_zero = True
                            break
                    
                    # If the zero is on the same line as a starred zero/ on the same line lying a starred zero, 
                    # cover the corresponding line, and uncover the column of the starred zero
                    if there_is_starred_zero:
                        covered_rows.add(row_indices[i])
                        covered_columns.remove(starred_zero_index[1])
                    else:
                        # If a non-covered zero has no assigned zero on its row, perform the following steps :
                        # Step 1: Find a starred zero on the corresponding column. 
                        # If there is one, go to Step 2, else, stop.
                        # Step 2: Find a primed zero on the corresponding row 
                        # (there should always be one). Go to Step 1.

                        # For all zeros encountered during the path, star 
                        # primed zeros and unstar starred zeros, remove all covered lines and primed zeros

                        # When this whole step is done, repeat from step 3.2, i.e. we don't search
                        # for non-covered zero anymore.

                        there_is_primed_zero = True

                        curr_index = np.array([row_indices[i], column_indices[i]])
                        while there_is_primed_zero:

                            starred_zero_index = np.zeros(2)
                            there_is_starred_zero = False
                            for j in range(kcomps):
                                # Check if there is a starred zero on this column.
                                if starred_matrix [j, curr_index[1]] == 1:
                                    starred_zero_index = [j, curr_index[1]]
                                    
                                    # Unstar this starred zero
                                    starred_matrix [j, curr_index[1]] = 0
                                    
                                    there_is_starred_zero = True
                                    break

                            if there_is_starred_zero:
                                # Star this primed zero
                                starred_matrix[curr_index[0], curr_index[1]] = 1
                                
                                for j in range(kcomps):
                                    # Check if there is a primed zero on this row (there should always be one)
                                    if (primed_matrix[starred_zero_index[0], j] == 1):
                                        curr_index = [starred_zero_index[0], j]
                                        break

                            else:  # If there is no starred zero on this column, stop.
                            
                                # Star this primed zero
                                starred_matrix[curr_index[0], curr_index[1]] = 1
                                
                                break

                        break

            all_zeros_are_covered = True
            # Check here whether all zeros are covered
            for i in range(kcomps):
                for j in range(kcomps):
                    if cost_matrix[i, j] == 0 and not ((i in covered_rows) or (j in covered_columns)):
                        all_zeros_are_covered = False

            # Stop when all zeros are covered and we now have the covered column and rows.
            if all_zeros_are_covered: 
                break
                        
            # Remove all covered line and primed zeros
            covered_columns = set()
            covered_rows = set()
            primed_matrix = np.zeros((kcomps, kcomps))
                        
        covered_columns = np.array(list(covered_columns))
        covered_rows = np.array(list(covered_rows))
        
        # If the number of covered lines equal k
        if (np.size(covered_columns) + np.size(covered_rows) == kcomps):

            # Here is possible to make an assignment
            assigned_cluster = np.apply_over_axes(np.argmin, cost_matrix, 0).reshape(kcomps)

            if np.size(np.unique(assigned_cluster)) < np.size(assigned_cluster):
                # If there is duplicate, try moving one of the overlapping downward or
                # upward until it works.

                for i in assigned_cluster:
                    # If there are more than one zeros on this row
                    if (np.count_nonzero(cost_matrix[i, :], 0) > 1):
                        
                        # Search for a row that has not yet been assigned
                        for j in range(kcomps):
                            if (j not in assigned_cluster):
                                assigned_cluster[i] = j
                
            assigned_cluster = assigned_cluster.reshape(kcomps, 1)
            indices = np.concatenate((assigned_cluster, np.arange(kcomps)[:, np.newaxis]), 1)

            cost = np.sum(origin_cost_matrix[indices[:,0],indices[:,1]])

            return cost

        # Step 4: From the elements that are left, find the lowest value. 
        # Subtract this from every unmarked element and add it to every element covered by two lines.

        temp_cost_matrix = copy.deepcopy(cost_matrix)

        for i in covered_rows:
            temp_cost_matrix[i, :] = np.inf
        for j in covered_columns:
            temp_cost_matrix[:, j] = np.inf

        min_val = np.argmin(temp_cost_matrix)

        sub_matrix = np.full((kcomps, kcomps), min_val)
        add_matrix = np.full((kcomps, kcomps), 0)

        for i in covered_rows:
            sub_matrix[i, :] = 0

        for j in covered_columns:
            add_matrix[:, j] = min_val

        cost_matrix = cost_matrix - sub_matrix
        cost_matrix = cost_matrix + add_matrix

if __name__ == '__main__':

    # Import and access the data
    iris = datasets.load_iris()
    iris_data = iris['data']

    # ===============  Aufgabe 1a ==========================

    # Repeat the algorithm if it fails to converge (i.e. determinant 0 because of computation error or
    # implementation error)
    while True:  # Rerun the algorithm when in some iteration the determinant equals 0
        try:
            means, cov_matrices, weights, resp_matrix = GMM(iris_data[:150, :], 3)  # Test on the first two entries
            break

        except np.linalg.LinAlgError:
            print("Matrix has 0 determinant")


    print("Means:\n", means)
    print("Covariance Matrices:\n", cov_matrices)
    print("weights:\n", weights)

    # ================== Aufgabe 1b ==============================================

    # Repeat the algorithm if it fails to converge (i.e. determinant 0 because of computation error or
    # implementation error)
    while True:  # Rerun the algorithm when in some iteration the determinant equals 0
        try:
            means, cov_matrices, weights, resp_matrix = GMM(iris_data[:150, :2], 3)  
            break

        except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
            print(e)


    # This code snippet is taken from the example from scikit.
    # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html
    colors = ["purple", "blue", "red"]

    h = plt.subplot(1, 1, 1)
    make_ellipses(means, cov_matrices, h)

    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(
            data[:, 0], data[:, 1], s=2.5, color=color, label=iris.target_names[n]
        )

    plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))

    # ============= Aufgabe 2 =============================

    n = iris_data.shape[0]
    # Run GMM or use the result from 1b
    
    # Repeat the algorithm if it fails to converge (i.e. determinant 0 because of computation error or
    # implementation error)
    while True:  # Rerun the algorithm when in some iteration the determinant equals 0
        try:
            means, cov_matrices, weights, resp_matrix = GMM(iris_data[:150, :], 3)  # Test on the first two entries
            break
        except np.linalg.LinAlgError:
            print("Matrix has 0 determinant")
    
    # Assign the point to the cluster that having the largest resposibility
    gmm_cluster_index = np.apply_over_axes(np.argmax, resp_matrix, 1).reshape(n)

    # The true classification / ground truth label
    target = iris.target

    # The accuracy of GMM
    cost_gmm = hungarian_matching(gmm_cluster_index, target)

    # Extract the cluster index of kmeans.
    kmeans = cluster.KMeans(n_clusters=3, random_state=0).fit(iris_data)
    kmean_cluster_index = kmeans.labels_

    # The accuracy of KMeans
    cost_kmeans = hungarian_matching(kmean_cluster_index, target)

    print("The amount of samples in the right class of GMM:", n-cost_gmm)
    print("The amount of samples in the right class of Kmeans:", n-cost_kmeans)


    plt.show()
