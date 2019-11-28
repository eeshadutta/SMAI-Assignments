import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

N = 100
d = 5
x = np.zeros((N, 1))
x[0:d] = 2*np.array(np.random.uniform(15, 27, (d, 1)))
mu = 0
sigma = 1
noise = np.random.normal(mu, sigma, N)
alpha = np.random.uniform(-0.01, 1, d)
s = np.sum(alpha)
alpha = alpha/s
alpha = np.reshape(alpha, (5, 1))

for i in range(5, N):
    x[i] = np.sum(np.multiply(alpha, x[i-d:i])) + \
        np.random.normal(mu, sigma, (1, 1))


plt.plot(x)
plt.ylabel("Data from Equation")
plt.savefig("3_1.png")
plt.show()


def gradient_descent(data, N, d):
    lr = 0.00000000001
    w = np.reshape(np.random.uniform(-0.01, 0.01, d), (d, 1))
    w /= np.sum(w)
    for i in range(d, N):
        s = np.sum(np.multiply(w, data[i-d:i]))
        err = data[i] - s
        w = w + lr * err * data[i-d:i]
    return w


pred = []
w = gradient_descent(x, N, d)
for i in range(d, N):
    pred.append(np.sum(np.multiply(w, x[i-d:i])))


plt.plot(x, label='Actual Data')
plt.plot(pred, label='Predicted Data')
plt.legend()
plt.title('Predicted vs Actual')
plt.savefig("3_2.png")
plt.show()


temp = np.arange(10)
err = np.zeros(temp.size)
for i in range(10):
    lr = 0.00000000000001
    w = np.reshape(np.random.uniform(-0.1, 0.5, temp[i]), (temp[i], 1))
    w /= np.sum(w)
    for j in range(temp[i], N):
        s = np.sum(np.multiply(w, x[j-temp[i]:j]))
        error = x[j] - s
        w = w + lr*error*x[j-temp[i]:j]
    for j in range(temp[i], N):
        err[i] += (x[j]-np.sum(np.multiply(w, x[j-temp[i]:j])))**2

# plt.plot(err, label='Error')
# plt.title('Error vs d')
# plt.savefig("3_3.png")
# plt.show()
