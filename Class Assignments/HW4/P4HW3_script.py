import numpy as np
import matplotlib.pyplot as plt
from scipy import random, linalg
from mpl_toolkits import mplot3d

nele = 1000
dim = 3

# Part - 1
# Two random mu, Sigma is Diagonal
mu1 = [0.5, 0.6, 0.2]
print('mu1 is ', mu1)
mu2 = [0.3, 0.2, 0.5]
print('mu2 is ', mu2)

sigma = [[0.6, 0, 0], [0, 0.4, 0], [0, 0, 0.5]]
print('sigma = ', sigma)

cl1 = np.random.multivariate_normal(mu1, sigma, (nele, nele))
cl2 = np.random.multivariate_normal(mu2, sigma, (nele, nele))

plt.figure()
plt.subplot(131).title.set_text('x-y plane')
plt.scatter(cl1[:, 0], cl1[:, 1], marker='o')
plt.scatter(cl2[:, 0], cl2[:, 1], marker='x')

plt.subplot(132).title.set_text('y-z plane')
plt.scatter(cl1[:, 1], cl1[:, 2], marker='o')
plt.scatter(cl2[:, 1], cl2[:, 2], marker='x')

plt.subplot(133).title.set_text('x-z plane')
plt.scatter(cl1[:, 0], cl1[:, 2], marker='o')
plt.scatter(cl2[:, 0], cl2[:, 2], marker='x')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(cl1[:, 0], cl1[:, 1], cl1[:, 2], marker='o')
ax.scatter(cl2[:, 0], cl2[:, 1], cl2[:, 2], marker='x')
ax.title.set_text('Data Plot')
plt.show()

# Part - 2
# Two random mu, sigma is PSD
mu3 = [0.2, 0.9, 0.4]
print('mu3 is ', mu3)
mu4 = [0.5, 0.1, 0.7]
print('mu4 is ', mu4)

sigma = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
print('sigma = ', sigma)

cl1 = np.random.multivariate_normal(mu3, sigma, (nele, nele))
cl2 = np.random.multivariate_normal(mu4, sigma, (nele, nele))

plt.figure()
plt.subplot(131).title.set_text('x-y plane')
plt.scatter(cl1[:, 0], cl1[:, 1], marker='o')
plt.scatter(cl2[:, 0], cl2[:, 1], marker='x')

plt.subplot(132).title.set_text('y-z plane')
plt.scatter(cl1[:, 1], cl1[:, 2], marker='o')
plt.scatter(cl2[:, 1], cl2[:, 2], marker='x')

plt.subplot(133).title.set_text('x-z plane')
plt.scatter(cl1[:, 0], cl1[:, 2], marker='o')
plt.scatter(cl2[:, 0], cl2[:, 2], marker='x')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(cl1[:, 0], cl1[:, 1], cl1[:, 2], marker='o')
ax.scatter(cl2[:, 0], cl2[:, 1], cl2[:, 2], marker='x')
ax.title.set_text('Data Plot')
plt.show()

# Part - 3
# Two Identical means, Two different sigmas
mu5 = [0.7, 0.6, 0.3]
print('mu5 is ', mu5)

sigma1 = [[0.6, 0, 0], [0, 0.4, 0], [0, 0, 0.5]]
sigma2 = [[0.3, 0, 0], [0, 0.1, 0], [0, 0, 0.8]]
print('sigma1 = ', sigma1)
print('sigma2 = ', sigma2)

cl1 = np.random.multivariate_normal(mu5, sigma1, (nele, nele))
cl2 = np.random.multivariate_normal(mu5, sigma2, (nele, nele))

plt.figure()
plt.subplot(131).title.set_text('x-y plane')
plt.scatter(cl1[:, 0], cl1[:, 1], marker='o')
plt.scatter(cl2[:, 0], cl2[:, 1], marker='x')

plt.subplot(132).title.set_text('y-z plane')
plt.scatter(cl1[:, 1], cl1[:, 2], marker='o')
plt.scatter(cl2[:, 1], cl2[:, 2], marker='x')

plt.subplot(133).title.set_text('x-z plane')
plt.scatter(cl1[:, 0], cl1[:, 2], marker='o')
plt.scatter(cl2[:, 0], cl2[:, 2], marker='x')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(cl1[:, 0], cl1[:, 1], cl1[:, 2], marker='o')
ax.scatter(cl2[:, 0], cl2[:, 1], cl2[:, 2], marker='x')
ax.title.set_text('Data Plot')
plt.show()
