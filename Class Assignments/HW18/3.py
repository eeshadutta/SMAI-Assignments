from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


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

req_train_data = np.empty((0, 784))
req_train_labels = np.empty((0))
req_test_data = np.empty((0, 784))
req_test_labels = np.empty((0))

for i in range(train_data.shape[0]):
    if train_labels[i] == 1:
        req_train_data = np.vstack([req_train_data, train_data[i]])
        req_train_labels = np.append(req_train_labels, 1)
    if train_labels[i] == 2:
        req_train_data = np.vstack([req_train_data, train_data[i]])
        req_train_labels = np.append(req_train_labels, -1)

for i in range(test_data.shape[0]):
    if test_labels[i] == 1:
        req_test_data = np.vstack([req_test_data, test_data[i]])
        req_test_labels = np.append(req_test_labels, 1)
    if test_labels[i] == 2:
        req_test_data = np.vstack([req_test_data, test_data[i]])
        req_test_labels = np.append(req_test_labels, -1)

clf = MLPClassifier(hidden_layer_sizes=(1000, 1000),
                    max_iter=100, learning_rate_init=0.0001)
clf.fit(req_train_data, req_train_labels)
plt.plot(clf.loss_curve_)
plt.title("Learning Curve")
plt.savefig("3_plot.pdf")
plt.show()

accuracy = clf.score(req_test_data, req_test_labels)
print(accuracy)
