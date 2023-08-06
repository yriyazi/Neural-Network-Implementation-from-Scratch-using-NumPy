import numpy as np

class SGD:
    '''
    Stochastic Gradient Descent (SGD) optimizer for neural network training.

    Args:
        learning_rate (float, optional): Learning rate for updating weights. Defaults to 1.0.
        decay (float, optional): Learning rate decay factor. Defaults to 0.0.
        momentum (float, optional): Momentum term for faster convergence. Defaults to 0.0.
        
    with help from: https://www.deeplearning.ai/ai-notes/optimization/index.html
    
    This optimizer implements the Stochastic Gradient Descent algorithm, which is a popular optimization technique used to update neural network parameters during training.
    It supports a learning rate that can be decayed over iterations and incorporates momentum to accelerate convergence.
    
    Methods:
        - `pre_decay()`: Applies learning rate decay if specified.
        - `mid_update_params(layer)`: Updates weights and biases of a neural network layer with or without momentum.
        - `post_itter()`: Updates iteration count after parameter updates.

    Note: This implementation is adapted from the original source: https://www.deeplearning.ai/ai-notes/optimization/index.html
    '''
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_decay(self):
        '''
        Apply learning rate decay if specified.
        '''
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def mid_update_params(self, layer):
        '''
        Update weights and biases of a neural network layer with or without momentum.
        
        Args:
            layer (Layer): The neural network layer to update.
        '''
        weight_updates = -self.current_learning_rate * layer.dweights
        bias_updates   = -self.current_learning_rate * layer.dbiases
        
        if self.momentum:
            weight_updates += self.momentum * layer.weight_momentums
            bias_updates   += self.momentum * layer.bias_momentums
            
            layer.weight_momentums  = weight_updates
            layer.bias_momentums    = bias_updates
            
        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_itter(self):
        '''
        Update iteration count after parameter updates.
        '''
        self.iterations += 1
