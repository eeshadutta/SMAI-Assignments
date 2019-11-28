import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

np.random.seed(100)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

print_flag = 0

class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):
    
        # :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        # :param int n_neurons: The number of neurons in this layer.
        # :param str activation: The activation function to use (if any).
        # :param weights: The layer's weights.
        # :param bias: The layer's bias.
        
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        self.activation = activation
        self.bias = bias if bias is not None else np.random.rand(n_neurons)

    def activate(self, x,bias):
    
        # Calculates the dot product of this layer.
        # :param x: The input.
        # :return: The result.

        if bias == True:
            r = np.dot(x, self.weights) + self.bias
        else:
            r = np.dot(x, self.weights)
        self.last_activation = self._apply_activation(r)
        return self.last_activation
    
    def _apply_activation(self, r):
        # Applies the chosen activation function (if any).
        # :param r: The normal value.
        # :return: The activated value.
 
        # In case no activation function was chosen
        if self.activation is None:
            return r
    
        # tanh
        if self.activation == 'tanh':
            return np.tanh(r)
    
        # sigmoid
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
    
        return r
    
    def apply_activation_derivative(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.

        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)

        return r

class NeuralNetwork:
    """
    Represents a neural network.
    """
 
    def __init__(self):
        self._layers = []
 
    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """
 
        self._layers.append(layer)

    def feed_forward(self, X,bias):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        for layer in self._layers:
            X = layer.activate(X,bias)

        return X

    # N.B: Having a sigmoid activation in the output layer can be interpreted 
    # as expecting probabilities as outputs.
    # W'll need to choose a winning class, this is usually done by choosing the 
    # index of the biggest probability.

    def backpropagation(self, X, y, learning_rate,bias):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        output = self.feed_forward(X,bias)

        # print(output)
        for i in range(len(output)):
            if output[i] >= 0.75:
                output[i] = 1
            else:
                output[i] = -1

        # Loop over the layers backward
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        
    
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate
    
    def train(self, X, y, learning_rate, max_epochs,bias):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """
 
        mses = []
 
        for i in range(max_epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate,bias)
            if i % 10 == 0:
                mse = np.mean(np.square(y - nn.feed_forward(X,bias)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
 
        return mses
 
    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """
 
        return (y_pred == y_true).mean()

    def predict(self, X,bias):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X,bias)
        # print(ff)
        # print(ff.shape)
        for i in range(len(ff)):
            if ff[i] >= 0.75:
                ff[i] = 1
            else:
                ff[i] = -1
        
        # One row
        # if ff.ndim == 1:
            # return np.argmax(ff)

        # Multiple rows
        # return np.argmax(ff, axis=1)
        return ff

## data creation
mean1 = [0, 0]
cov1 = [[1, 0], [0, 100]]

mean2 = [2,1]
cov2 = [[4,0],[0,9]]

x1,y1 = np.random.multivariate_normal(mean1,cov1,100).T
x2,y2 = np.random.multivariate_normal(mean2,cov2,100).T

# data = np.empty([200,200])
labels = np.empty([200])

train_data = np.empty([2,160])
train_labels = np.empty([160])
test_data = np.empty([2,40])
test_labels = np.empty([40])

for i in range(80):
    train_data[0][2*i] = x1[i]
    train_data[0][2*i+1] = x2[i]
    train_data[1][2*i] = y1[i]
    train_data[1][2*i+1] = y2[i]
    train_labels[2*i] = 0
    train_labels[2*i+1] = 1

for i in range(20):
    test_data[0][2*i] = x1[80+i]
    test_data[0][2*i+1] = x2[80+i]
    test_data[1][2*i] = y1[80+i]
    test_data[1][2*i+1] = y2[80+i]
    test_labels[2*i] = 0
    test_labels[2*i+1] = 1

if print_flag == 1:
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.scatter(x1,y1,marker='x')
    ax1.scatter(x2,y2,marker='o')
    ax2.scatter(train_data[0][0::2],train_data[1][0::2],marker='x')
    ax2.scatter(train_data[0][1::2],train_data[1][1::2],marker='o')
    ax3.scatter(test_data[0][0::2],test_data[1][0::2],marker='x')
    ax3.scatter(test_data[0][1::2],test_data[1][1::2],marker='o')
    plt.show()


nn = NeuralNetwork()
nn.add_layer(Layer(2, 2, 'tanh'))
nn.add_layer(Layer(2, 1, 'sigmoid'))
 
# # print(train_data.shape)
errors = nn.train(train_data.T,train_labels,0.003,200,False)
print(nn.predict(test_data.T,False))
print(test_labels)
print('Accuracy: %.2f%%' % (nn.accuracy(nn.predict(test_data.T,False), test_labels.flatten()) * 100))



clf = MLPClassifier(hidden_layer_sizes=(2),max_iter=200,solver='adam',activation='tanh',learning_rate_init=0.1)
clf.fit(train_data.T, train_labels)
plt.plot(errors,label="Without Bias")
plt.plot(clf.loss_curve_,label='Inbuilt Implementation')
plt.legend()
plt.show()
y_pred = clf.predict(test_data.T)
print("Acc",clf.score(test_data.T,test_labels))
print("Accuracy of MLPClassifier 1 hidden layer with tanh activation :", accuracy_score(test_labels,y_pred)*100)