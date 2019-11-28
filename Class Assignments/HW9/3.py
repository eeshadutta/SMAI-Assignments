import numpy as np
import matplotlib.pyplot as plt

mu_dist = 0
var_dist = 1


def mle_change(s, k):
    mean = np.empty(0)
    variance = np.empty(0)
    for i in range(s):
        dist = np.random.normal(mu_dist, var_dist, k)
        mu_set = np.mean(dist)
        var_set = np.var(dist)
        mean = np.append(mean, mu_set)
        variance = np.append(variance, var_set)

    return mean, variance


mean, variance = mle_change(100, 100)
plt.plot(mean)
plt.ylabel("Mean")
plt.title("s estimates of mean")
plt.show()

kvar = np.empty(0)
for k in range(1, 100):
    mean, variance = mle_change(100, k)
    m = np.var(mean)
    kvar = np.append(kvar, m)
plt.plot(kvar)
plt.title("Variance of the set with change in k (s=100)")
plt.ylabel("Variance")
plt.xlabel("k")
plt.show()

kvar = np.empty(0)
for s in range(1, 100):
    mean, variance = mle_change(s, 100)
    m = np.var(mean)
    kvar = np.append(kvar, m)
plt.plot(kvar)
plt.title("Variance of the set with change in s (k=100)")
plt.ylabel("Variance")
plt.xlabel("s")
plt.show()
