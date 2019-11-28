import numpy as np
import matplotlib.pyplot as plt

normal_dist1 = np.random.normal(10, 20, 1000)
normal_dist2 = np.random.normal(20, 30, 1000)
samples = normal_dist1 + normal_dist2
w = np.linspace(-2000, 2000)

loss_lin = []
loss_log = []


def sigmoid(x):
    return 1/(1 + np.exp(-x))


for i in range(len(w)):
    l1 = l2 = 0
    for j in range(len(samples)):
        if j >= 1000:
            l1 = l1 + (-1-w[i]*samples[j])**2
            l2 = l1 + (-1-sigmoid(w[i]*samples[j]))**2
        else:
            l1 = l1+(1-w[i]*samples[j])**2
            l2 = l2+(1-sigmoid(w[i]*samples[j]))**2
    loss_lin.append(l1)
    loss_log.append(l2)

plt.subplot(1, 2, 1)
plt.plot(w, loss_lin)
plt.title("Linear Regression")
plt.subplot(1, 2, 2)
plt.plot(w, loss_log)
plt.title("logistic Regression")

plt.suptitle("Loss vs w")
plt.savefig("Loss.png")
plt.show()
