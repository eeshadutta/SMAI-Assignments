from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

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
        data[ind] = [ int(x) for x in num[1:] ]
        
    return (data, labels)

train_data, train_labels = read_data("sample_train.csv")
test_data, test_labels = read_data("sample_test.csv")
print(train_data.shape, test_data.shape)
print(train_labels.shape, test_labels.shape)

batch_size = train_data.shape[0]
X = train_data[0:batch_size,:]
mean = np.mean(X, axis = 0)
print(mean.shape)
var = np.var(X, axis = 0)
for i in range(X.shape[1]):
    X[:,i] = (X[:,i]-mean[i])
dim_to_red = 2
rand_in = np.random.randn(X.shape[1],dim_to_red)
v_L2,_ = np.linalg.qr(rand_in)
v_L1,_ = np.linalg.qr(rand_in)

def sgn(x):
    output = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (x[i,j]>0):
                output[i,j] = 1
            elif(x[i,j]<0):
                output[i,j] -1
            else:
                output[i,j] = 0
    return output

max_iter = 50
lr = 0.01
reg = 0.1
threshold = int(1e-10)
for iters in range(max_iter):
    t0 = np.matmul(np.matmul(X,v_L2),v_L2.T) - X
    t1 = np.matmul(np.matmul(v_L2,v_L2.T),X.T) - (-X).T
    t0_norm = np.linalg.norm(t0)
    t1_norm = np.linalg.norm(t1)
    derv_L2 = np.matmul(np.matmul(X.T,t0),v_L2)/t0_norm + np.matmul(np.matmul(t1,X),v_L2)/t1_norm +  reg*(v_L2)/np.linalg.norm(v_L2)
    derv_L1 = np.matmul(np.matmul(X.T,t0),v_L2)/t0_norm + np.matmul(np.matmul(t1,X),v_L2)/t1_norm +  reg*sgn(v_L1)
    v_L2_old = v_L2
    v_L2 -= lr*derv_L2
    v_L2, _ = np.linalg.qr(v_L2)
    v_L1_old = v_L1
    v_L1 -= lr*derv_L1
    v_L1, _ = np.linalg.qr(v_L1)
    improve = np.linalg.norm(np.abs(v_L2_old-v_L2))
    if (improve < threshold):
        break

PCA_gdL2_data = np.matmul(X,-v_L2)
data = X
sigma = np.cov(data.T)
eigvals, eigvecs = np.linalg.eig(sigma)

idx = eigvals.argsort()[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:,idx]
PCA_data = (np.matmul(data,eigvecs[:,idx[0:dim_to_red]]))

fig = plt.figure(figsize = [15,15], dpi = 60)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.scatter(-PCA_gdL2_data[:,0], PCA_gdL2_data[:,1], marker='o', c = 'red', label = 'Class 1')
ax1.set_title('GD-PCA with L2 regularization', fontsize = 40)
ax2.scatter(PCA_data[:,0], PCA_data[:,1], marker='o', c = 'red', label = 'Class 1')
ax2.set_title('PCA', fontsize = 40)
plt.savefig('smaihw11q3.jpg')
plt.show()