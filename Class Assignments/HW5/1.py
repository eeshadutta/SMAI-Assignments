import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import svm

ele = 1000

mu1 = np.array([3, 3])
mu2 = np.array([7, 7])

sigma1 = np.array([[3, 0], [0, 3]])
sigma2 = np.array([[3, 0], [0, 3]])

# sigma1 = np.array([[3, 1], [2, 3]])
# sigma2 = np.array([[7, 2], [1, 7]])

cl1 = np.random.multivariate_normal(mu1, sigma1, ele) % 10
cl2 = np.random.multivariate_normal(mu2, sigma2, ele) % 10

fig = plt.figure()
plt.scatter(cl1[:, 0], cl1[:, 1], marker='o', color='red')
plt.scatter(cl2[:, 0], cl2[:, 1], marker='x', color='green')
plt.show() 

