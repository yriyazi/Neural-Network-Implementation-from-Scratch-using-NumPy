import numpy as np 
from utils.one_hot import convertToOneHot
from utils.configuration import num_classes

class MSE():
    
    def forward(self,y_hat, y,num_classes=10):
        """
        booth y_hat and y must be one-hotted and divide to batch size
        
        """
        if len(y.shape) == 1:
            y = convertToOneHot(y, num_classes)
        
        return np.sum((y_hat-y)**2)/y_hat.shape[0]
    
    def backward(self,y_hat, y,num_classes=10):
        """
        Computes mean squared error gradient between targets and divide the result by the batch size.
        """
        if len(y.shape) == 1:
            y = convertToOneHot(y, num_classes)
            
        self.dinputs = 2*(y_hat-y)/y_hat.shape[0]

    
    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.L2 > 0:
            regularization_loss += layer.L2 * np.sum(layer.weights * layer.weights)
            regularization_loss += layer.L2 * np.sum(layer.biases  * layer.biases)

        return regularization_loss