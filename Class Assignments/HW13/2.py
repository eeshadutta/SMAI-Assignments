import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    num_points = len(lines)
    dim_points = 28 * 28
    data = np.empty((num_points, dim_points))
    labels = np.empty(num_points)

    for ind, line in enumerate(lines):
        num = line.split(',')
        labels[ind] = int(num[0])
        data[ind] = [int(x) for x in num[1:]]

    return (data, labels)


train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)

X = train_data - np.mean(train_data, axis=0)
X = X / np.var(X)
Y = (np.arange(np.max(train_labels)+1) == train_labels[:, None]).astype(float)
testX = test_data - np.mean(test_data, axis=0)
testX = testX / np.var(testX)
testY = test_labels
num_class = len(np.unique(Y))


def derivative(w):
    prob = []
    probs = np.exp(np.matmul(w, X.T))
    total_prob = np.sum(probs, axis=0)
    for i in range(10):
        p = ((probs[i, :]/total_prob) - Y.T[i]).T @ X
        prob.append(p)
    prob = np.array(prob)
    return prob


def gradient_descent():
    num_iters = 100
    rate = 0.1
    threshold = 1e-10
    w = np.random.uniform(size=(10, 784))
    for _ in range(num_iters):
        w_old = w
        w = w_old - rate*derivative(w_old)
        if np.linalg.norm(w - w_old) < threshold:
            break
    return w


def predict(w):
    probs = np.exp(np.matmul(w, testX.T))
    total_prob = np.sum(probs, axis=0)
    cl = np.argmax(probs/total_prob, axis=0)
    return cl


w_optimal = gradient_descent()
ans = predict(w_optimal)
num_ele = testY.shape[0]
acc = 0
for i in range(num_ele):
    if ans[i] == testY[i]:
        acc += 1
acc = acc / num_ele
acc = acc * 100
print("Accuracy =", acc)
