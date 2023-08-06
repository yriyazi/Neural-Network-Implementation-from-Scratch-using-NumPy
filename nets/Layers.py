import numpy as np
import utils

from utils.configuration import mu,sigma,bias

# https://www.deeplearning.ai/ai-notes/initialization/index.html
# https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
# Dense layer
class Dense:

    def __init__(self, n_inputs, n_neurons,
                 mean=utils.mu,variance=utils.sigma,bias=utils.bias,
                 L2=0 ):
        
        # Initialize weights and biases
        if utils.Xavier==True:
            self.weights = np.random.normal(mean,(variance/n_neurons),[n_inputs, n_neurons])+bias
        else:
            self.weights = np.random.normal(mean,(variance),[n_inputs, n_neurons])+bias#0.01 * np.random.randn(n_inputs, n_neurons)
            
        self.biases = np.random.uniform(-1, 1, size=(1, n_neurons))*variance

        # Set regularization strength
        self.L2  = L2
        
        #setting momentum
        self.weight_momentums  = np.zeros_like(self.weights)
        self.bias_momentums    = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        if self.L2 > 0:
            self.dweights += 2 * self.L2 *  self.weights
            self.dbiases  += 2 * self.L2 *  self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout
#https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
class Dropout():
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate,size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
