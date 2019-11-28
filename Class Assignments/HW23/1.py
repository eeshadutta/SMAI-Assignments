import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from copy import deepcopy

X = np.array([[-1, 1], [-1, -1], [1, 1], [1, -1],
              [0, 1], [0, -1], [0, 0.5], [0, -0.5]])

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(X[:, 0], X[:, 1])
ax.set_title("Data points")


init_method = [(-0.5, 0), (0.5, 0)]
kmeans_model = KMeans(n_clusters=2, init=np.asarray(
    init_method), n_init=1).fit(X)
centers = kmeans_model.cluster_centers_
labels = kmeans_model.labels_
ax = fig.add_subplot(122)
ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
ax.scatter(centers[:, 0], centers[:, 1], color='black')
ax.set_title("K-Means clustering")

plt.show()
