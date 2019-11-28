import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0,0],[3,0],[4,2],[1,2]])
parallelogram = np.array([[0,0],[3,0],[4,2],[1,2],[0,0]])
mu = np.array([np.mean(x[:,0]),np.mean(x[:,1])])
print('mean = ',mu)

sigma = np.cov(x.T)
print('Covariance = \n', sigma)

eigvals, eigvecs = np.linalg.eig(sigma)

print('Eigenvectors are = \n', eigvecs)
fig = plt.figure(figsize = [20,20], dpi = 40)
ax = fig.add_subplot(111)
ax.plot(parallelogram[:,0], parallelogram[:,1])
ax.scatter(x[:,0],x[:,1], marker = 'x', color = 'red', s = 500)
m = [mu[0]],[mu[1]]
ax.quiver(*m, eigvecs[:,0], eigvecs[:,1], color = ['green','black'], scale = 5)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.tick_params(axis='both', which='minor', labelsize=50)
ax.set_xlim([-5,10])
ax.set_ylim([-5,10])
plt.show()