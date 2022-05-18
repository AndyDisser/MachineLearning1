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


if __name__ == "__main__":
    # ============= Aufgabe 2 =============================

    # Import and access the data
    iris = datasets.load_iris()
    iris_data = iris['data']
    
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