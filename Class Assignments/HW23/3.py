from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score


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


data, labels = read_data("sample_test.csv")

color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'cyan',
             5: 'violet', 6: 'orange', 7: 'brown', 8: 'navy', 9: 'aqua'}
colors = [color_map[int(i)] for i in labels]

# PCA
pca = PCA(n_components=2)
X = pca.fit_transform(data)


# Ground Truth plot
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dataplot after PCA')
plt.savefig('3_PCA.png')
plt.show()


def kmeans_clustering(data, k, ninit=10, init_method='k-means++'):
    kmeans = KMeans(n_clusters=k, init=init_method, n_init=ninit).fit(data)
    return kmeans


def goodness_of_clusters(ground_truth, results):
    score = completeness_score(ground_truth, results) * 100
    return score


# K-means plot
kmeans_model = kmeans_clustering(X, 10)
colors = [color_map[int(i)] for i in kmeans_model.labels_]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Dataplot after K-Means clustering')
plt.savefig('3_Kmeans.png')
plt.show()

# Goodness Score
goodness = goodness_of_clusters(labels, kmeans_model.labels_)
print('Goodness Measure of Clustering =', goodness)

# Kmeans for 5 different random initialization
scores = []
plt.figure(figsize=(19, 9))
for i in range(5):
    kmeans_model = kmeans_clustering(X, 10, 1, 'random')
    kmeans_labels = kmeans_model.labels_
    goodness_score = goodness_of_clusters(labels, kmeans_labels)
    scores.append(goodness_score)
    print('Goodness Measure with Random Initialization', goodness_score)
    colors = [color_map[int(i)] for i in kmeans_labels]
    if i == 0:
        loc = (0, 0)
    elif i == 1:
        loc = (0, 2)
    elif i == 2:
        loc = (0, 4)
    elif i == 3:
        loc = (1, 1)
    elif i == 4:
        loc = (1, 3)
    ax = plt.subplot2grid((2, 6), loc, colspan=2)
    ax.scatter(X[:, 0], X[:, 1], c=colors)

plt.suptitle('K-Means Clustering with Random Initialization')
plt.savefig('3_Random.png')
plt.show()

# K-Means initialized with Ground Truth
ground_truth_centroid = []
for label in range(10):
    data_class = [data[i]
                  for i in range(data.shape[0]) if int(labels[i]) == label]
    x_sum = 0
    y_sum = 0
    for dc in data_class:
        x_sum += dc[0]
        y_sum += dc[1]
    centroid = (x_sum / len(data_class), y_sum / len(data_class))
    ground_truth_centroid.append(centroid)

kmeans_gt_model = kmeans_clustering(
    X, 10, 1, np.asarray(ground_truth_centroid))
kmeans_gt_labels = kmeans_gt_model.labels_
goodness_score = goodness_of_clusters(labels, kmeans_gt_labels)
print('Goodness Measure with Ground Truth Initialization =', goodness_score)
print('Number of Iterations required =', kmeans_gt_model.n_iter_)

colors = [color_map[int(i)] for i in kmeans_gt_labels]
plt.scatter(X[:, 0], X[:, 1], c=colors)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means Clustering with Gound Truth Initialization')
plt.savefig('3_GT.png')
plt.show()
