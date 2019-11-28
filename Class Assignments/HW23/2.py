import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

num_samples = 10
x = np.random.randint(0, 40, num_samples)
y = np.random.randint(0, 40, num_samples)
data = [(x[i], y[i]) for i in range(10)]
print("Dataset:", data)


def kmeans_clustering(data, k):
    kmeans = KMeans(n_clusters=k).fit(data)
    labels = kmeans.labels_
    inertia = kmeans.inertia_
    return labels, inertia


inertias = []
for k in range(num_samples):
    labels, inertia = kmeans_clustering(data, k+1)
    inertias.append(inertia)


idx = np.arange(1, 11)
plt.plot(idx, inertias)
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Inertia vs K')
plt.savefig('2_plot.png')
plt.show()
