from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)

    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [int(x) for x in num[1:]]

    return (data, labels)


train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
train_data = train_data-np.mean(train_data, axis=0)
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)

cov = np.cov(train_data, rowvar=False)
eig_vals, eig_vecs = LA.eig(cov)
eig_vecs = eig_vecs[:, 0:2]
train_data_PCA = np.matmul(train_data, eig_vecs)


w = np.random.rand(train_data.shape[1], 2)
w, _, _ = LA.svd(w)
w = w[:, :2]
num_iter = 50
learning_rate = 0.01
threshold = 1e-10

while num_iter > 0:
    A = np.matmul(np.matmul(train_data, w), w.T)-train_data
    B = np.matmul(np.matmul(w, w.T), train_data.T)-(-train_data).T
    der = (np.matmul(np.matmul(train_data.T, A), w))/LA.norm(A) + \
        (np.matmul(np.matmul(B, train_data), w))/LA.norm(B)
    w_old = w
    w -= learning_rate * der
    w, _ = LA.qr(w)
    check = LA.norm(np.abs(w_old - w))
    if (check < threshold):
        break
    num_iter -= 1

plt.figure(figsize=[15, 15], dpi=60)
plt.scatter(train_data_PCA[:, 0:1], train_data_PCA[:, 1:2])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Normal PCA')
plt.savefig("PCA.png")

train_data_PCA_GD = np.matmul(train_data, -w)
plt.figure(figsize=[15, 15], dpi=60)
plt.scatter(train_data_PCA_GD[:, 0:1], -train_data_PCA_GD[:, 1:2])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('GD PCA')
plt.savefig("GD_PCA.png")
