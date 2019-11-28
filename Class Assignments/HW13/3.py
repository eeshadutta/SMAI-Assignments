import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sn


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

X = train_data
Y = np.array(train_labels).astype('int64')
testX = test_data
testY = np.array(test_labels).astype('int64')
num_class = len(np.unique(Y))

classifiers = {}
for i in range(num_class):
    for j in range(i+1, num_class):
        clf = LogisticRegression(
            solver='liblinear', multi_class='ovr', max_iter=80)
        fit = clf.fit(np.concatenate((X[Y == i, :], X[Y == j, :]), axis=0), np.concatenate(
            (Y[Y == i], Y[Y == j]), axis=0))
        pred = fit.predict(testX)
        classifiers[i, j] = np.array(pred)

class_pred = np.zeros(testY.shape)

for i in range(testY.shape[0]):
    votes = np.zeros((num_class, 1))
    for j in range(num_class):
        for k in range(j+1, num_class):
            pred = classifiers[j, k]
            votes[pred[i]] += 1
    class_pred[i] = np.argmax(votes)

num_ele = len(testY)
acc = 0
for i in range(num_ele):
    if class_pred[i] == testY[i]:
        acc += 1
acc = acc / num_ele
acc = acc * 100
print("Accuracy =", acc)

con_mat = confusion_matrix(testY, class_pred)
plt.figure(figsize=(15, 15))
sn.heatmap(con_mat, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title('Confusion Matrix')
plt.savefig('3_ConfusionMatrix.png')
plt.show()
