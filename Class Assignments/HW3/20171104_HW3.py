import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

x = [0, 1, 2, 1000]
y = [0, 1, 2, 1000]

plt.plot(x, y, color='black', linewidth=1)

data_xc = []
data_yc = []
for i in range(0, 1000):
    data_xc.append(i)
    j = i + random.uniform(-500, 500)
    data_yc.append(j)

x_mean = 0
y_mean = 0
for i in range(0, 1000):
    x_mean = x_mean + data_xc[i]
    y_mean = y_mean + data_yc[i]
x_mean = x_mean / 1000
y_mean = y_mean / 1000

data = np.stack((data_xc, data_yc), axis=0)
cov_data = np.cov(data)

eig_val, eig_vec = LA.eig(cov_data)

origin = [x_mean], [y_mean]
plt.quiver(*origin, eig_vec[:, 0], eig_vec[:, 1], color=['r', 'g'], scale=3)

plt.scatter(data_xc, data_yc, color='blue')

plt.show()
