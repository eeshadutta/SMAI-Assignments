import numpy as np
import matplotlib.pyplot as plt

data = np.array([[2, 2, 1], [-1, 1, 1], [1, -1, 1],
                 [-1, -1, 1], [-2, -2, 1], [1, 1, 1]])
labels = np.array([1, 1, 1, -1, -1, -1])
J = []

w = [1, 0, -1]
mc_set = []
j = 0
for i in range(0, 6):
    val = labels[i] * np.matmul(w, data[i].T)
    if val < 0:
        mc_set.append(i)
        j += val
J.append(-j)

rate = 1
num_iter = 20
ni = 0
while len(mc_set) != 0:
    old_w = w
    for m in mc_set:
        der = labels[m] * data[m]
    w = old_w + rate * der
    mc_set.clear()
    j = 0
    for i in range(0, 6):
        val = labels[i] * np.matmul(w, data[i].T)
        if val < 0:
            mc_set.append(i)
            j += val
    J.append(-j)
    ni += 1
    if ni == num_iter:
        break

print(J)

x = np.arange(0, len(J))
plt.plot(x, J, '-o')
plt.show()
