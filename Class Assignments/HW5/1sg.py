import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp
n = 1000

mu1 = np.array([3, 3])
mu2 = np.array([7, 7])

sigma = 3*np.eye(2)
sigma1 = np.array([[3, 1], [2, 3]])
sigma2 = np.array([[7, 2], [1, 7]])


def distribution_val(mu, sigma, x, d=2):

    val_denom = (((2*pi)**(d/2))*(np.linalg.det(sigma)**(0.5)))
    sigma_inverse = np.linalg.inv(sigma)
    xsigmaxT = np.matmul(np.matmul((x-mu), sigma_inverse), np.transpose(x-mu))
    val_num = exp(-(0.5*xsigmaxT))
    return (val_num/val_denom).real


testdata = np.random.uniform(0, 11, [n, 2])
label1 = np.zeros(testdata.shape[0])
label2 = np.zeros(testdata.shape[0])
# Part 1
for i in range(testdata.shape[0]):
    testval_dist1 = distribution_val(mu1, sigma, testdata[i])
    testval_dist2 = distribution_val(mu2, sigma, testdata[i])
    if (testval_dist1 > testval_dist2):
        label1[i] = 1
    else:
        label1[i] = 2

# Part 2
for i in range(testdata.shape[0]):
    testval_dist1 = distribution_val(mu1, sigma1, testdata[i])
    testval_dist2 = distribution_val(mu2, sigma2, testdata[i])
    if (testval_dist1 > testval_dist2):
        label2[i] = 1
    else:
        label2[i] = 2

fig = plt.figure(figsize=[15, 15])
ax1 = fig.add_subplot(121)
plt.scatter(testdata[label1 == 1, 0],
            testdata[label1 == 1, 1], marker='o', color='red')
plt.scatter(testdata[label1 == 2, 0],
            testdata[label1 == 2, 1], marker='x', color='green')
ax2 = fig.add_subplot(122)
plt.scatter(testdata[label2 == 1, 0],
            testdata[label2 == 1, 1], marker='o', color='red')
plt.scatter(testdata[label2 == 2, 0],
            testdata[label2 == 2, 1], marker='x', color='green')
plt.show()
