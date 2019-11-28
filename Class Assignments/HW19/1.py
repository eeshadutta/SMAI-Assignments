import numpy as np
import matplotlib.pyplot as plt

num_points = 100
cl1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_points)
cl2 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_points)
cl1_labels = np.ones(num_points)
cl2_labels = np.zeros(num_points)

train_data = np.concatenate((cl1[0:80, :], cl2[0:80, :]), axis=0)
test_data = np.concatenate((cl1[80:100, :], cl2[80:100, :]), axis=0)
train_label = np.concatenate((cl1_labels[0:80], cl2_labels[0:80]))
test_label = np.concatenate((cl1_labels[80:100], cl2_labels[80:100]))


def sig(z):
    return 1/(1+np.exp(-z))


def gradient_descent(X, Y, num_iter, lr, w1, w2):
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

    acc = 0
    l1 = np.tanh(np.matmul(test_data, w1))
    y1 = sig(np.matmul(l1, w2.T))
    for i in range(40):
        if y1[i] >= 0.5:
            if test_label[i] == 1:
                acc += 1
        else:
            if test_label[i] == 0:
                acc += 1

    return(loss, acc)


o1 = gradient_descent(train_data, train_label, 1000,
                      1, np.zeros((2, 2)), np.zeros(2))
o2 = gradient_descent(train_data, train_label, 1000,
                      1, np.ones((2, 2)), np.ones(2))
o3 = gradient_descent(train_data, train_label, 1000, 1,
                      np.random.uniform(-1, 1, size=(2, 2)), np.random.uniform(-1, 1, 2))

print("Accuracy with zero weight = ", o1[1]/40)
print("Accuracy with one weight = ", o2[1]/40)
print("Accuracy with random weight = ", o3[1]/40)

plt.plot(np.array(o1[0]), label='Zero weight')
plt.plot(np.array(o2[0]), label='One weight')
plt.plot(np.array(o3[0]), label='Random weight')
plt.title('Learning Curve')
plt.legend()
plt.savefig('1_plots.png')
plt.show()
