import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn import svm


def decision_boundary(x):
    return ((1.92*x + 11.54) - np.sqrt((1.92*x + 11.54) - 4*0.67*(205-1.23*x*x-38.43*x))) / 1.34


ele = 7
mu_vec1 = np.array([12/7, 8/7])
mu_vec2 = np.array([54/7, 60/7])

x1_samples = np.array([[0, 0], [0, 1], [2, 0], [3, 2], [3, 3], [2, 2], [2, 0]])
x2_samples = np.array(
    [[7, 7], [8, 6], [9, 7], [8, 10], [7, 10], [8, 9], [7, 11]])

mu_vec1 = mu_vec1.reshape(1, 2).T
mu_vec2 = mu_vec2.reshape(1, 2).T

f, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x1_samples[:, 0], x1_samples[:, 1],
           marker='o', color='green', s=40, alpha=0.5)
ax.scatter(x2_samples[:, 0], x2_samples[:, 1],
           marker='^', color='blue', s=40, alpha=0.5)
plt.legend(['Class1 (w1)', 'Class2 (w2)'], loc='upper right')
plt.ylabel('x2')
plt.xlabel('x1')
ftext = 'p(x|w1) ~ N(mu_vec1=(0,0)^t, cov1=I)\np(x|w2) ~ N(mu_vec2=(1,1)^t, cov2=I)'
plt.figtext(.15, .8, ftext, fontsize=11, ha='left')

x_1 = np.arange(-30, 10, 0.1)
bound = decision_boundary(x_1)
plt.plot(x_1, bound, 'r--', lw=3)

x_vec = np.linspace(*ax.get_xlim())
x_1 = np.arange(0, 100, 0.05)

plt.show()
