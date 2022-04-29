# This is a sample Python script.
import matplotlib.pyplot as plt
import numpy as np
# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

lNumbers = np.random.random_sample(1000)
# x = np.array([1.2, 10.0, 12.4, 15.5, 20.])
# print(x)
# bins = np.array([0, 5, 10, 15, 20])
# print(bins)
# print(np.digitize(x,bins,right=True))
def avg_of_two(lst):
    avg_list = []
    for i in range(1,len(lst),2):
        avg_list.append(np.mean([lst[i-1], lst[i]]))
    return avg_list

def avg_of_x(lst, x):
    avg_list = []
    while len(lst) >= x:
        intermediate_list = []
        for _ in range(x):
            intermediate_list.append(lst.pop(0))
        avg_list.append(np.mean(intermediate_list))
    return avg_list

def histogram_nk(avg_over_numbers=2, number_of_runs=1000):
    avg_lst = []
    for _ in range(number_of_runs):
        avg_lst.append(np.mean(np.random.random_sample(avg_over_numbers)))
    plt.hist(avg_lst, bins=20)
    plt.show()


histogram_nk(5)
histogram_nk(10)
histogram_nk(100)
# lst_avg = []
# for i in range(0,len(lNumbers),2):
#     lst_avg.append((lNumbers[i] + lNumbers[i+1]) / 2)
#
# plt.hist(lst_avg)
# plt.show()

