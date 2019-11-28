import numpy as np
import matplotlib.pyplot as plt

mu1 = np.array([-1, 0])
sigma1 = np.array([[2, 0], [0, 2]])
mu2 = np.array([1, 0])
sigma2 = np.array([[3, 0], [0, 3]])
num_points = 100
cl1 = np.random.multivariate_normal(mu1, sigma1, num_points)
cl2 = np.random.multivariate_normal(mu2, sigma2, num_points)
cl1_labels = np.zeros(num_points)
cl2_labels = np.ones(num_points)

train_data = np.concatenate((cl1[0:80, :], cl2[0:80, :]), axis=0)
test_data = np.concatenate((cl1[80:100, :], cl2[80:100, :]), axis=0)
train_label = np.concatenate((cl1_labels[0:80], cl2_labels[0:80]))
test_label = np.concatenate((cl1_labels[80:100], cl2_labels[80:100]))

w1 = np.random.randn(3, 2)
w2 = np.random.randn(5, 3)
w3 = np.random.randn(3, 5)
w4 = np.random.randn(1, 3)


def sig(x):
    return 1/(1+np.exp(-x))


def sig_prime(x):
    return x*(1-x)


def feedforward(x):
    x1 = sig(w1.dot(x))
    x2 = sig(w2.dot(x1))
    x3 = sig(w3.dot(x2))
    x4 = sig(w4.dot(x3))
    return x4


def backprop(x, y, y_pred, lr, loss_type):
    global w1, w2, w3, w4
    sz = y.shape[0]
    grad0 = 0
    grad1 = 0
    grad2 = 0
    grad3 = 0
    for i in range(sz):
        x0 = x[i, :].T
        x1 = sig(w1.dot(x0))
        x2 = sig(w2.dot(x1))
        x3 = sig(w3.dot(x2))
        err = y[i] - y_pred[i]
        # if loss_type == 'MSE':
        #     err = y[i] - y_pred[i]
        # else:
        #     err = -y[i]
        del4 = err * sig_prime(y_pred[i])
        grad3 += del4 * x3
        del3 = del4 * w4.T * (sig_prime(x3)[:, None])
        grad2 += del3.dot(x2[:, None].T)
        del2 = w3.T.dot(del3) * (sig_prime(x2)[:, None])
        grad1 += del2.dot(x1[:, None].T)
        del1 = w2.T.dot(del2) * (sig_prime(x1)[:, None])
        grad0 += del1.dot(x0[:, None].T)
    w4 += lr * grad3
    w3 += lr * grad2
    w2 += lr * grad1
    w1 += lr * grad0


def loss(y, y_pred):
    return 0.5 * np.sum((y-y_pred)**2)/(y.shape[0])


def hinge_loss(y, y_pred):
    l = 0
    for i in range(y.shape[0]):
        l += max(0, 1-y[i]*y_pred[i])
    return l


def neural_network(lr, epochs, loss_type):
    n = train_label.shape[0]
    y_pred = np.zeros(n)
    loss_plot = np.zeros(epochs)
    for iter in range(epochs):
        for i in range(n):
            y_pred[i] = feedforward(train_data[i, :].T)
        if (loss_type == 'MSE'):
            loss_plot[iter] = loss(train_label, y_pred)
        else:
            loss_plot[iter] = hinge_loss(train_label, y_pred)
        backprop(train_data, train_label, y_pred, lr, loss_type)

    acc = 0
    for i in range(test_data.shape[0]):
        acc += ((feedforward(test_data[i, :].T) >= 0.5) == test_label[i])
    acc = acc / test_data.shape[0]
    return loss_plot, acc


loss_plot_mse, acc1 = neural_network(0.01, 1000, 'MSE')
print('Accuracy with MSE loss =', acc1)
loss_plot_hinge, acc2 = neural_network(0.01, 1000, 'hinge')
print('Accuracy with hinge loss =', acc2)

fig = plt.figure(figsize=[15, 10], dpi=60)
ax1 = fig.add_subplot(121)
ax1.plot(loss_plot_mse, label='MSE loss')
ax1.legend()
ax2 = fig.add_subplot(122)
ax2.plot(loss_plot_hinge, label='Hinge loss')
ax2.legend()
plt.suptitle('Loss Curve')
plt.savefig('2_plots.png')
plt.show()
