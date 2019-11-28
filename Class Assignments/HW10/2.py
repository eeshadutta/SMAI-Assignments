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
        data[ind] = [round(float(x), 2) for x in num[1:]]

    return (data, labels)


def eigen_decomposition(data):
    cov = np.cov(data, rowvar=False)
    eig_vals, eig_vecs = LA.eig(cov)
    x = np.arange(1, 14)
    plt.bar(x, eig_vals)
    plt.xticks(np.arange(0, 14, 1))
    plt.xlabel("Number of eigen values")
    plt.ylabel("Madnitude of eigen value")
    plt.title("Eigen Values")
    plt.show()
    return eig_vals, eig_vecs


train_data, train_labels = read_data("wine.csv")
eig_vals, eig_vecs = eigen_decomposition(train_data)
pc1 = eig_vecs[:, 0]
pc2 = eig_vecs[:, 1]

cl1_x = []
cl1_y = []
cl2_x = []
cl2_y = []
cl3_x = []
cl3_y = []

for i in range(train_data.shape[0]):
    if train_labels[i] == 1:
        cl1_x.append(np.matmul(train_data[i], pc1))
        cl1_y.append(np.matmul(train_data[i], pc2))
    elif train_labels[i] == 2:
        cl2_x.append(np.matmul(train_data[i], pc1))
        cl2_y.append(np.matmul(train_data[i], pc2))
    else:
        cl3_x.append(np.matmul(train_data[i], pc1))
        cl3_y.append(np.matmul(train_data[i], pc2))

plt.scatter(cl1_x, cl1_y, marker='1', color='r')
plt.scatter(cl2_x, cl2_y, marker='2', color='g')
plt.scatter(cl3_x, cl3_y, marker='3', color='b')
plt.legend(['class1', 'class2', 'class3'], loc='upper right')
plt.title("Scatter plot")
plt.show()
