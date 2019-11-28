import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Perceptron

data = np.array([[1, 1], [2, 4], [-2, 1], [-3, -1]])
classes = np.array([1, 1, -1, -1])
x_min = data[:, 0].min() - 1
x_max = data[:, 0].max() + 1
y_min = data[:, 1].min() - 1
y_max = data[:, 1].max() + 1
diff = 0.005
xx, yy = np.meshgrid(np.arange(x_min, x_max, diff),
                     np.arange(y_min, y_max, diff))

gamma = 1

fig = plt.figure(figsize=(40, 40), dpi=60)

clfn_perc = Perceptron()
fit_perc = clfn_perc.fit(data, classes)
ax1 = fig.add_subplot(211)
pred_perc = fit_perc.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax1.contourf(xx, yy, pred_perc)
ax1.scatter(data[classes == 1, 0], data[classes == 1, 1],
            marker='x', color='red', s=500)
ax1.scatter(data[classes == -1, 0], data[classes == -1, 1],
            marker='o', color='blue', s=500)
ax1.set_title('Perceptron')

clfn_lr = LogisticRegression(solver='lbfgs', multi_class='auto')
fit_lr = clfn_lr.fit(gamma*data, classes)
ax2 = fig.add_subplot(212)
pred_lr = fit_lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax2.contourf(xx, yy, pred_lr)
ax2.scatter(data[classes == 1, 0], data[classes == 1, 1],
            marker='x', color='red', s=500)
ax2.scatter(data[classes == -1, 0], data[classes == -1, 1],
            marker='o', color='blue', s=500)
ax2.set_title('Logistic Regression')

plt.savefig('1_compare.png')
plt.show()


fig = plt.figure(figsize=(50, 50), dpi=60)
clfn_lr = LogisticRegression(solver='lbfgs', multi_class='auto')
i = 1
loop = 4
for m in range(loop, 11*loop, loop):
    gamma = m / 10
    fit_lr = clfn_lr.fit(gamma*data, classes)
    plt.subplot(5, 2, i)
    pred_lr = fit_lr.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contourf(xx, yy, pred_lr)
    plt.scatter(data[classes == 1, 0], data[classes == 1, 1],
                marker='x', color='red', s=500)
    plt.scatter(data[classes == -1, 0], data[classes == -1, 1],
                marker='o', color='blue', s=500)
    plt.title('Logistic Regression: gamma = %f' % (gamma))
    i += 1
plt.savefig('1_gammacomparison.png')
plt.show()
