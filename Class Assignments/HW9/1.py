import numpy as np
import matplotlib.pyplot as plt
pure = np.linspace(-3, 1, 1000)
noise = np.random.normal(0, 1, pure.shape)
data = pure + noise
x = np.arange(1, 1001)
plt.scatter(x, data)
plt.show()


def calculate_huber_loss(a, delta, theta0, theta1):
    loss = 0
    for i in range(0, 1000):
        if(abs(a[i]-theta1*x[i]-theta0) <= delta):
            loss += ((a[i]-theta1*x[i]-theta0)*(a[i]-theta1*x[i]-theta0))/2
        else:
            loss += delta*(a[i]-theta1*x[i]-theta0)-(delta*delta)/2
    return loss


def update_huber_loss(a, delta, theta0, theta1, rate):
    sum1 = 0
    sum2 = 0
    for i in range(0, 1000):
        if abs(a[i]-theta1*x[i]-theta0 <= delta):
            sum1 += -(a[i]-theta1*x[i]-theta0)
            sum2 += -(a[i]-theta1*x[i]-theta0)*x[i]
        else:
            if a[i]-theta1*x[i]-theta0 >= 0:
                sum1 += -delta
                sum2 += -delta*x[i]
            else:
                sum1 += delta
                sum2 += delta*x[i]
    theta0 = theta0-rate*sum1
    theta1 = theta1-rate*sum2
    return theta0, theta1


def calculate_logcosh(a, theta0, theta1):
    return np.sum(np.log(np.cosh(theta0+theta1*x - a)))


def update_logcosh(a, theta0, theta1, rate):
    sum1 = 0
    sum2 = 0
    for i in range(0, 1000):
        sum1 += np.tanh(-a[i]+theta1*x[i]+theta0)
        sum2 += np.tanh(-a[i]+theta1*x[i]+theta0)*x[i]
    theta0 = theta0-rate*sum1
    theta1 = theta1-rate*sum2
    return theta0, theta1


def calculate_quantile_loss(a, gama, theta0, theta1):
    loss = 0
    for i in range(0, 1000):
        if(a[i] < theta0+x[i]*theta1):
            loss += (gama-1)*(theta0+x[i]*theta1-a[i])
        else:
            loss += gama*(a[i]-theta0-x[i]*theta1)
    return loss


def update_quantile_loss(a, gama, theta0, theta1, rate):
    sum1 = 0
    sum2 = 0
    for i in range(0, 1000):
        if(a[i] < theta0+x[i]*theta1):
            sum1 += (gama-1)
            sum2 += (gama-1)*x[i]
        else:
            sum1 += -gama
            sum2 += -gama*x[i]
    theta0 = theta0-rate*sum1
    theta1 = theta1-rate*sum2
    return theta0, theta1


def gradient_descent(a, type_loss):
    delta = 0.9
    the1 = 1
    the2 = 1
    rate = 8*1e-8
    iterations = 1000
    gama = 2.04
    loss = np.zeros(iterations)
    slope = np.zeros(iterations)
    i = 0
    while(iterations):
        if type_loss == 'huber':
            loss[i] = abs(calculate_huber_loss(a, delta, the1, the2))
            the1, the2 = update_huber_loss(a, delta, the1, the2, rate)
        elif type_loss == 'logcosh':
            loss[i] = abs(calculate_logcosh(a, the1, the2))
            the1, the2 = update_logcosh(a, the1, the2, rate)
        elif type_loss == 'quantile':
            loss[i] = abs(calculate_quantile_loss(a, gama, the1, the2))
            the1, the2 = update_quantile_loss(a, gama, the1, the2, rate)
        slope[i] = -the2/the1
        i += 1
        iterations -= 1
    x_new = np.arange(1, 1001)
    plt.scatter(x_new, loss)
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.title(type_loss)
    plt.show()
    plt.scatter(x_new, slope)
    plt.xlabel('iterations')
    plt.ylabel('Slope')
    plt.title(type_loss)
    plt.show()


gradient_descent(data, 'huber')
gradient_descent(data, 'logcosh')
gradient_descent(data, 'quantile')
