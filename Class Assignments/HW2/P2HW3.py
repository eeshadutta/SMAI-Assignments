import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

A = [[0, 0, 1] for i in range(50)]
B = [[0, 0, 1] for i in range(50)]
fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(50):
    A[i][0] = round(random.random() * 2 - 1, 2)
    A[i][1] = round(random.random() * 2 - 1, 2)
    B[i][0] = round(random.random() * 2 - 1, 2)
    B[i][1] = round(random.random() * 2 - 1, 2)
    ax.scatter(A[i][0], A[i][1], 1, marker='x', c='tab:red')
    ax.scatter(B[i][0], B[i][1], 1, marker='o', c='tab:blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')

w = [[0, 0, 0] for i in range(5)]
w[0] = [1, 1, 0]
w[1] = [-1, -1, 0]
w[2] = [0, 0.5, 0]
w[3] = [1, -1, 5]
w[4] = [1.0, 1.0, 0.3]

x = np.linspace(-1, 1, 100)
y = (-w[0][2] - x*w[0][0])/(w[0][1])
ax.plot(x, y, 'g')
plt.title('Plot for A')
plt.show()

# Accuracy checking
for j in range(5):
    n = 0
    for i in range(50):
        temp = w[j][0]*A[i][0] + w[j][1]*A[i][1] + w[j][2]*A[i][2]
        if (temp > 0):
            n = n + 1
    for i in range(50):
        temp = w[j][0]*B[i][0] + w[j][1]*B[i][1] + w[j][2]*B[i][2]
        if (temp <= 0):
            n = n + 1
    print("Accuracy for", w[j], "=", n, "%")
