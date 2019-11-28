import numpy as np
import matplotlib.pyplot as plt

num_iter = 50


def Gradient_Descent(n):
    prev = -2
    i = 1
    err = np.empty(0)
    while i <= num_iter:
        err = np.append(err, prev * prev)
        curr = prev - n * 2 * prev
        i = i + 1
        prev = curr
    return err


axes = plt.gca()
axes.set_xlim([0, 10])
axes.set_ylim([0, 100])
err1 = Gradient_Descent(0.1)
err2 = Gradient_Descent(1.5)
err3 = Gradient_Descent(1)
plt.plot(err1, color='g')
# plt.show()
plt.plot(err2, color='r')
# plt.show()
plt.plot(err3, color='b')
plt.legend(['n=0.1', 'n=1.5', 'n=1'], loc='upper right')
plt.show()
