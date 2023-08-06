import numpy as np
import utils

from utils.configuration import mu, sigma, bias

class Dense:
    '''
    Dense layer for a neural network.

    Args:
        n_inputs (int): Number of input neurons.
        n_neurons (int): Number of neurons in the layer.
        mean (float, optional): Mean value for weight initialization. Defaults to utils.mu.
        variance (float, optional): Variance value for weight initialization. Defaults to utils.sigma.
        bias (float, optional): Bias value for weight initialization. Defaults to utils.bias.
        L2 (float, optional): L2 regularization strength. Defaults to 0.

    with help from: [https://www.deeplearning.ai/ai-notes/initialization/index.html,
                     https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/]
                     
    This class represents a dense (fully connected) layer in a neural network. It includes methods for forward and backward passes, weight initialization, and regularization.

    Methods:
        - `forward(inputs)`: Performs forward pass through the layer.
        - `backward(dvalues)`: Performs backward pass through the layer to compute gradients.
    '''

    def __init__(self, n_inputs, n_neurons, mean=utils.mu, variance=utils.sigma, bias=utils.bias, L2=0):
        '''
        Initialize the dense layer.

        Args:
            n_inputs (int): Number of input neurons.
            n_neurons (int): Number of neurons in the layer.
            mean (float, optional): Mean value for weight initialization. Defaults to utils.mu.
            variance (float, optional): Variance value for weight initialization. Defaults to utils.sigma.
            bias (float, optional): Bias value for weight initialization. Defaults to utils.bias.
            L2 (float, optional): L2 regularization strength. Defaults to 0.
        '''
        if utils.Xavier == True:
            self.weights = np.random.normal(mean, (variance / n_neurons), [n_inputs, n_neurons]) + bias
        else:
            self.weights = np.random.normal(mean, (variance), [n_inputs, n_neurons]) + bias
        self.biases = np.random.uniform(-1, 1, size=(1, n_neurons)) * variance

        self.L2 = L2

        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.biases)

    def forward(self, inputs):
        '''
        Perform forward pass through the layer.

        Args:
            inputs (numpy.ndarray): Input data.

        This method computes the output of the layer given the input data.
        '''
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        '''
        Perform backward pass through the layer to compute gradients.

        Args:
            dvalues (numpy.ndarray): Gradient values from the previous layer.

        This method computes the gradients of the layer's weights, biases, and inputs.
        '''
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.L2 > 0:
            self.dweights += 2 * self.L2 * self.weights
            self.dbiases += 2 * self.L2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

class Dropout:
    '''
    Dropout layer for neural network regularization.

    Args:
        rate (float): Dropout rate (probability of dropping a neuron's output).
        
    with help from: https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html

    This class implements the Dropout regularization technique in a neural network.
    Dropout randomly sets a fraction of neuron outputs to zero during each forward pass, reducing overfitting.
    
    Methods:
        - `forward(inputs)`: Applies dropout during the forward pass.
        - `backward(dvalues)`: Computes gradients during the backward pass.
    '''

    def __init__(self, rate):
        '''
        Initialize the Dropout layer.

        Args:
            rate (float): Dropout rate (probability of dropping a neuron's output).
        '''
        self.rate = 1 - rate

    def forward(self, inputs):
        '''
        Apply dropout during the forward pass.

        Args:
            inputs (numpy.ndarray): Input data.

        This method applies dropout to the input data, randomly setting a fraction of neuron outputs to zero.
        '''
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        '''
        Compute gradients during the backward pass.

        Args:
            dvalues (numpy.ndarray): Gradient values from the next layer.

        This method computes the gradients for the backward pass by applying the same binary mask as during the forward pass.
        '''
        self.dinputs = dvalues * self.binary_mask
