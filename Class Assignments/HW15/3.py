import numpy as np
from sklearn import svm
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib import cm

x = np.array([[1, 1], [-1, -1], [-1, 1], [1, -1]])
y = np.array([1, 1, -1, -1])


def plot_boundary(ax, c, k):
    clf = svm.SVC(C=c, kernel=k, gamma='auto')
    clf = clf.fit(x, y)
    h = 0.01
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))
    pred = clf.predict(np.c_[xx1.ravel(), xx2.ravel()])
    pred = pred.reshape(xx1.shape)
    ax.pcolormesh(xx1, xx2, pred, cmap=cm.Accent)
    ax.scatter(x[y == 1, 0], x[y == 1, 1], color='blue',
               marker='.', s=200, label='Class 1')
    ax.scatter(x[y == -1, 0], x[y == -1, 1], color='red',
               marker='x', s=100, label='Class 2')
    ax.set_title(k+' kernel and C = '+str(c), fontsize=16)


c = np.array([1/125, 1, 125]).astype('float')
k = ['linear', 'rbf', 'sigmoid']
itr = 1
fig = plt.figure(figsize=[20, 20], dpi=80)
for i in range(3):
    for j in range(3):
        ax = fig.add_subplot(3, 3, itr)
        plot_boundary(ax, c[j], k[i])
        itr += 1
plt.savefig('plot.png')
plt.show()
