from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    num_points = len(lines)
    dim_points = 13
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)
    
    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [ round(float(x),2) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("wine.csv")

def eign_decomposition(data_matrix):

    cov_mtx = np.cov(data_matrix,rowvar=False)
    eig_vals ,eig_vecs = LA.eig(cov_mtx)
    print(eig_vecs.shape)
    np.set_printoptions(suppress=True)
    idx = list(range(1,14))
    print(eig_vals)
    plt.bar(idx,eig_vals)
    plt.xticks(np.arange(0, 14, 1))
    plt.xlabel('Number of eign values')
    plt.ylabel('Magnitudes')
    plt.title('Sorted Eign Values')
    plt.show()

    return eig_vals,eig_vecs


eig_vals,eig_vecs = eign_decomposition(train_data)
vec1 = eig_vecs[:,0]
vec2 = eig_vecs[:,1]
x_cors_1 = []
y_cors_1 = []

x_cors_2 = []
y_cors_2 = []

x_cors_3 = []
y_cors_3 = []

print(train_data.shape[0])

for i in range(train_data.shape[0]):

    if train_labels[i] == 1:
        x_cors_1.append(np.matmul(train_data[i],vec1))
        y_cors_1.append(np.matmul(train_data[i],vec2))
    elif train_labels[i] == 2:
        x_cors_2.append(np.matmul(train_data[i],vec1))
        y_cors_2.append(np.matmul(train_data[i],vec2))
    elif train_labels[i] == 3:
        x_cors_3.append(np.matmul(train_data[i],vec1))
        y_cors_3.append(np.matmul(train_data[i],vec2))


# x_cors = np.matmul(train_data,vec1)
# y_cors = np.matmul(train_data,vec2)
# print(len(x_cors_1))
plt.scatter(x_cors_1,y_cors_1,marker=',',color='r')
plt.scatter(x_cors_2,y_cors_2,marker='+',color='g')
plt.scatter(x_cors_3,y_cors_3,marker='D',color='b')
plt.show()

