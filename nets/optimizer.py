import numpy as np

# https://www.deeplearning.ai/ai-notes/optimization/index.html
class SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_decay(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def mid_update_params(self, layer):
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
        self.iterations += 1