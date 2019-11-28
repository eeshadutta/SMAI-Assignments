import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions


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


train_data, train_labels = read_data('sample_train.csv')
test_data, test_labels = read_data('sample_test.csv')
train_data = train_data - np.mean(train_data, axis=0) / np.var(train_data)
test_data = test_data - np.mean(test_data, axis=0) / np.var(test_data)

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

pca = PCA(n_components=2)
req_train_data = pca.fit_transform(req_train_data)
req_test_data = pca.fit_transform(req_test_data)
req_train_labels = req_train_labels.astype(int)
req_test_labels = req_test_labels.astype(int)

C = 1e-3
clf = SVC(C=C, kernel='linear')
fit = clf.fit(req_train_data, req_train_labels)
pred = fit.predict(req_test_data)
print("C =", C, ": Accuracy =", metrics.accuracy_score(req_test_labels, pred)*100)
color = ['red' if cl == 1 else 'green' for cl in req_train_labels]
plt.scatter(req_train_data[:, 0], req_train_data[:, 1], color=color)
plot_decision_regions(req_train_data, req_train_labels, clf=clf)
plt.title('Decision Boundary')
plt.savefig("Decision_Boundary.png")
plt.show()

color = ['red' if cl == 1 else 'green' for cl in req_train_labels]
plt.scatter(req_train_data[:, 0], req_train_data[:, 1], color=color)
plot_decision_regions(req_train_data, req_train_labels, clf=clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[
            :, 1], facecolors='none', s=100, zorder=30, edgecolors='k')
plt.title('Support Vectors marked in black')
plt.savefig('Support_Vectors.png')
plt.show()
