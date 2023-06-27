REGULARIZATION

0-l2_reg_cost.py:
Contains a function l2_reg_cost(cost, lambtha, weights, L, m) for calculating the cost of a neural network with L2 regularization. The L2 regularization cost is calculated using the formula:

J_reg = cost + (lambtha / (2 * m)) * sum(weights^2)

where:
cost is the original cost function
lambtha is the regularization parameter
m is the number of data points used
weights are the weights of the neural network

The function includes the descriptions:
cost: cost of the network without L2 regularization
lambtha: regularization parameter
weights: dictionary of the weights and biases (numpy.ndarrays) of the neural network
L: number of layers in the neural network
m: number of data points used
The function returns the cost of the network accounting for L2 regularization.

1-l2_reg_gradient_descent.py:
Contains a function l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L) for updating the weights and biases of a neural network using gradient descent with L2 regularization. The weights are updated using the formula:

weights = weights - alpha * (dW + (lambtha / m) * weights)
biases = biases - alpha * db

where:

weights are the weights of the neural network
alpha is the learning rate
dW is the gradient of the weights
lambtha is the regularization parameter
m is the number of data points used
db is the gradient of the biases
The function takes the following parameters:

Y: one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
weights: dictionary of the weights and biases of the neural network
cache: dictionary of the outputs of each layer of the neural network
alpha: learning rate
lambtha: L2 regularization parameter
L: number of layers of the network
