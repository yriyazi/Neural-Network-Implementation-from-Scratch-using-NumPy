import numpy as np
from nets.Activation_functions  import Softmax
from utils.one_hot              import convertToOneHot
from utils.configuration        import num_classes

class Loss_CategoricalCrossentropy():
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confiDenses = y_pred_clipped[range(len(y_pred)),y_true]

        elif len(y_true.shape) == 2:
            correct_confiDenses = np.sum(y_pred_clipped * y_true,axis=1)
            
        self.output=np.mean(-np.log(correct_confiDenses))
        return np.mean(-np.log(correct_confiDenses))

    def backward(self, dvalues, y_true):
        if len(y_true.shape) == 1:
            y_true =convertToOneHot(y_true,num_classes)
            
        self.dinputs = np.divide(np.divide(-y_true,dvalues),len(dvalues))

    def regularization_loss(self, layer):
        regularization_loss = 0

        if layer.L2 > 0:
            regularization_loss += layer.L2 * np.sum(layer.weights * layer.weights)
            regularization_loss += layer.L2 * np.sum(layer.biases  * layer.biases)

        return regularization_loss