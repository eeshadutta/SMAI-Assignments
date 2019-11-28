import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

num_points = 100
cl1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_points)
cl2 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_points)
cl1_labels = np.ones(num_points)
cl2_labels = np.zeros(num_points)

train_data = np.concatenate((cl1[0:80, :], cl2[0:80, :]), axis=0)
test_data = np.concatenate((cl1[80:100, :], cl2[80:100, :]), axis=0)
train_label = np.concatenate((cl1_labels[0:80], cl2_labels[0:80]))
test_label = np.concatenate((cl1_labels[80:100], cl2_labels[80:100]))


def sig(x):
    return 1/(1+np.exp(-x))


def gradient_descent(X, Y, num_iter, lr):
    w1 = np.random.uniform(0, 1, size=(2, 2))
    w2 = np.random.uniform(0, 1, 2)
    loss = []
    while num_iter > 0:
        X2 = np.tanh(np.matmul(X, w1))
        X3 = sig(np.matmul(X2, w2.T))
        l = np.sum((Y-X3)**2) / 160
        loss.append(l)

        der2 = 0
        for i in range(160):
            der2 += (Y[i] - X3[i]) * X3[i] * (1 - X3[i]) * X2[i, :]
        w2 += 2 * lr * der2/160

        der1 = 0
        for i in range(160):
            der1 += (Y[i]-X3[i])*X3[i]*(1-X3[i]) * \
                np.array((X[i, :]*(w2[0] * (1-X2[i, 0]**2)),
                          X[i, :]*(w2[1] * (1-X2[i, 1]**2)))).T
        w1 += 2 * lr * der1/160

        num_iter -= 1

    return(loss, w1, w2)


ret1 = gradient_descent(train_data, train_label, 100, 1)
x2 = np.tanh(np.matmul(test_data, ret1[1]))
x3 = sig(np.matmul(x2, ret1[2].T))
# L1 = np.sum((test_label - x3)**2 / 40)
acc = 0
for i in range(len(x3)):
    if x3[i] > 0.5:
        if test_label[i] == 1:
            acc += 1
    else:
        if test_label[i] == 0:
            acc += 1
A1 = acc * 2.5


def gradient_descent_bias(X, Y, num_iter, lr):
    w1 = np.random.uniform(0, 1, size=(3, 2))
    w2 = np.random.uniform(0, 1, 3)
    loss = []
    t_data = np.array([X[:, 0], X[:, 1], np.ones(160)]).T
    while num_iter > 0:
        X2 = np.vstack((np.tanh(np.matmul(t_data, w1)).T, np.ones(160))).T
        X3 = sig(np.matmul(X2, w2.T))
        l = np.sum((Y-X3)**2) / 160
        loss.append(l)

        der2 = 0
        for i in range(160):
            der2 += (Y[i] - X3[i]) * X3[i] * (1 - X3[i]) * X2[i, :]
        w2 += 2 * lr * der2/160

        der1 = 0
        for i in range(160):
            der1 += (Y[i] - X3[i]) * X3[i] * (1 - X3[i]) * np.vstack((np.array((X[i, :] *
                                                                                (w2[0] * (1-X2[i, 0]**2)), X[i, :]*(w2[1] * (1-X2[i, 1]**2)))).T, np.ones(2)))
        w1 += 2 * lr * der1/160

        num_iter -= 1

    return(loss, w1, w2)


ret2 = gradient_descent_bias(train_data, train_label, 100, 1)
new_test = np.array([test_data[:, 0], test_data[:, 1], np.ones(40)]).T
x2 = np.vstack((np.tanh(np.matmul(new_test, ret2[1])).T, np.ones(40))).T
x3 = sig(np.matmul(x2, ret2[2].T))
# L2 = np.sum((test_label - x3)**2 / 40)
acc = 0
for i in range(len(x3)):
    if x3[i] > 0.5:
        if test_label[i] == 1:
            acc += 1
    else:
        if test_label[i] == 0:
            acc += 1
A2 = acc * 2.5

print('Accuracy without Bias =', A1)
print('Accuracy with Bias =', A2)

clf = MLPClassifier(hidden_layer_sizes=(
    2, ), max_iter=100, learning_rate_init=1)
clf.fit(train_data, train_label)
acc = clf.score(test_data, test_label)
print('Accuracy of Inbuilt =', acc*100)

plt.plot(np.array(ret1[0]), label='Without Bias')
plt.plot(np.array(ret2[0]), label='With Bias')
plt.plot(clf.loss_curve_, label='Inbuilt')
plt.legend()
plt.title("Learning Curve")
# plt.savefig('2_plots.png')
plt.show()
