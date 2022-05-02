"""
Name - Matrikelnummer 
1) Pham, Ngoc Anh Trung - 7176267
2) Viktor Vironski - 4330455
3) Andy Disser - 5984875

Exercise Sheet 1
"""

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def min_max_scale(data, min_range=0, max_range=1):
    """
    Scale data using min-max-scaling.
    
    Input: an array containing data.
    Return the the according array containing the scaled data, the max value, the min value.
    """
    
    max_val_dataset = np.amax(data)  # Max value of dataset.
    min_val_dataset = np.amin(data)  # Min value of dataset.

    # Define a function for scaling the data.
    min_max_scale_func = lambda x : (x - min_val_dataset) / (max_val_dataset - min_val_dataset)

    # Apply the function to the data array and return it, as well as the max and min value of the dataset.
    return min_max_scale_func(data), max_val_dataset, min_val_dataset

def z_score_normalize(data):
    """
    Normalize data using z score.

    Input: an array containing data.
    Return the the according array containing the scaled data, the mean value, the deviation.
    """

    # Compute mean
    mean = np.sum(data) / np.size(data)

    func = lambda x : np.power((x-mean), 2)
    
    # Compute deviation
    dev = np.sqrt( np.sum(func(data)) / np.size(data) )

    # Define a function for scaling the data according to z score method.
    z_score_scale_function = lambda x : (x-mean) / dev

    return z_score_scale_function(data), mean, dev


def random_three_plot(title, x_axis_label, y_axis_label):

    values1 = np.random.default_rng().uniform(1,5,100)  # Create 100 random numbers in [1,5]
    values2 = np.random.default_rng().uniform(5,6,100)  # Create 100 random numbers in [5,6]
    values3 = np.random.default_rng().uniform(6,10,100)  # Create 100 random numbers in [6,10]

    x = np.linspace(1, 100, num=100)  # Create the array: [1,2,...,100]

    fig1, myplot = plt.subplots(figsize=(10,10))  # Define the plot.

    myplot.set_title(title)                 # Set title
    myplot.set_xlabel(x_axis_label)         # Set x axis title
    myplot.set_ylabel(y_axis_label)         # Set y axis title
    myplot.plot(x, values1, 'b--', label="blue dashed")
    myplot.plot(x, values2, color='green', marker='o', mec='m', markersize=3, label="green line with magneta circle marker")
    myplot.plot(x, values3, color='red', linestyle='dotted', marker='^', mec='black', markersize=3, label="red dotted with black upper triangle marker")
    myplot.legend(shadow=True, fontsize='x-small')          # Add legend.
    
    #plt.show()

def compare_iris_data2d(data, min_max_scaled_data_a, min_max_scaled_data_b):

    f, axarr = plt.subplots(1,3, figsize=(8,10))        # Define the plot axes.
    
    im1 = axarr[0].imshow(data[:30])                    # Plot the first image.
    axarr[0].set_title("Before scaling")
    # Making the colorbar standing next to the graph.
    divider = make_axes_locatable(axarr[0])
    cax = divider.append_axes('right', size='15%', pad=0.05)
    f.colorbar(im1, cax=cax, orientation='vertical')
    
    im2 = axarr[1].imshow(min_max_scaled_data_a[:30])   # Plot the second image.
    axarr[1].set_title("After scaling (version a)")
    # Making the colorbar standing next to the graph.
    divider = make_axes_locatable(axarr[1])
    cax = divider.append_axes('right', size='15%', pad=0.05)
    f.colorbar(im2, cax=cax, orientation='vertical')

    im3 = axarr[2].imshow(min_max_scaled_data_b[:30])   # Plot the second image.
    axarr[2].set_title("After scaling (version b)")
    # Making the colorbar standing next to the graph.
    divider = make_axes_locatable(axarr[2])
    cax = divider.append_axes('right', size='15%', pad=0.05)
    f.colorbar(im3, cax=cax, orientation='vertical')

    #plt.show()

def compare_iris_scatter(data, z_score_scaled_data_a):

    f3, axarr = plt.subplots(1,2, figsize=(10,5))        # Define the plot axes.
    axarr[0].scatter(data[:, 0], data[:, 1], s=10)      # Define the first plot.
    axarr[0].set_xlabel("Sepal length")
    axarr[0].set_ylabel("Sepal width")
    axarr[0].set_title("Before scaling")

    axarr[1].scatter(z_score_scaled_data_a[:, 0], z_score_scaled_data_a[:, 1], s=10)        # Define the second plot.
    axarr[1].set_xlabel("Sepal length")
    axarr[1].set_ylabel("Sepal width")
    axarr[1].set_title("After scaling (version a)")

    #plt.show()

if __name__ == '__main__':

    # Import and access the data
    iris = datasets.load_iris()
    iris_data = iris['data']
    
    # ========= Aufgabe 1 ==========================================
    # a) Scale all the rows separately and print the results.
    print("Aufg 1a)===================================================")
    min_max_scaled_data_a = np.zeros((150, 4))

    # Scale 150 rows separately
    for i in range(150):
        min_max_scaled_data_a[i], max_val, min_val = min_max_scale(iris_data[i])

    print("The first 5 original rows")
    print(iris_data[:5])
    print("The first 5 scaled rows")
    print(min_max_scaled_data_a[:5])

    # b) Scale the whole database and print the first 5 features.
    print("\nAufg 1b)===================================================")
    min_max_scaled_data_b, max_val, min_val = min_max_scale(iris_data)
    
    print("The first 5 original rows")
    print(iris_data[:5])
    print("The first 5 scaled rows")
    print(min_max_scaled_data_b[:5])
    
    # ========= Aufgabe 2 ==========================================
    # a) Scale all the columns separately and print the first 5 rows.
    print("\nAufg 2a)===================================================")
    z_score_normalized_data_a = np.zeros((150, 4))

    # Scale the 4 columns separately
    for j in range(4):
        z_score_normalized_data_a[:,j], mean, dev = z_score_normalize(iris_data[:,j])
    
    print("The first 5 original rows")
    print(iris_data[:5])
    print("The first 5 scaled rows")
    print(z_score_normalized_data_a[:5])

    print("\nAufg 2b)===================================================")
    # b) Scale the whole dataset and print the first 5 rows.
    z_score_normalized_data_b = np.zeros((150, 4))
    z_score_normalized_data_b, mean, dev = z_score_normalize(iris_data)
    
    print("The first 5 original rows")
    print(iris_data[:5])
    print("The first 5 scaled rows")
    print(z_score_normalized_data_b[:5])

    
    # Aufgabe 3a
    random_three_plot("My plot", "xaxis", "yaxis")
    
    # Aufgabe 3b
    compare_iris_data2d(iris_data, min_max_scaled_data_a, min_max_scaled_data_b)
    
    # Aufgabe 3c
    compare_iris_scatter(iris_data, z_score_normalized_data_a)

    plt.show()  # Show the three figures of exercise 3.
