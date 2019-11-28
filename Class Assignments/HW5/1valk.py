import matplotlib.pyplot as plt
import numpy as np
import random


# distribution 1
mean1 = [3, 3]

# distribution 2
mean2 = [7, 7]


# #1
# cov_1 = [[3,0],[0,3]]
# cov_2 = [[3,0],[0,3]]


# x=[]
# y=[]
# while True:
#     b = np.random.multivariate_normal(mean1,cov_1,1)
#     x_x = b[0][0]
#     y_y = b[0][1]
#     if 0 < x_x and x_x <10:
#         if 0<y_y and y_y <10:
#             x.append(x_x)
#             y.append(y_y)
#     if len(x) == 1000:
#         break
# plt.scatter(x,y)


# x=[]
# y=[]
# while True:
#     b = np.random.multivariate_normal(mean2,cov_2,1)
#     x_x = b[0][0]
#     y_y = b[0][1]
#     if 0 < x_x and x_x <10:
#         if 0<y_y and y_y <10:
#             x.append(x_x)
#             y.append(y_y)
#     if len(x) == 1000:
#         break
# plt.scatter(x,y,color='red')
# plt.show()


# 2
cov_1 = [[3, 1], [2, 3]]
cov_2 = [[7, 2], [1, 7]]


x = []
y = []
while True:
    b = np.random.multivariate_normal(mean1, cov_1, 1)
    x_x = b[0][0]
    y_y = b[0][1]
    if 0 < x_x and x_x < 10:
        if 0 < y_y and y_y < 10:
            x.append(x_x)
            y.append(y_y)
    if len(x) == 1000:
        break
plt.scatter(x, y)


x = []
y = []
while True:
    b = np.random.multivariate_normal(mean2, cov_2, 1)
    x_x = b[0][0]
    y_y = b[0][1]
    if 0 < x_x and x_x < 10:
        if 0 < y_y and y_y < 10:
            x.append(x_x)
            y.append(y_y)
    if len(x) == 1000:
        break
plt.scatter(x, y, color='red')
plt.show()
