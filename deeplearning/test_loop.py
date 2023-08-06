import numpy as np
import utils
from deeplearning.train_loop import train

class test(train):
    def load(self):
        for lay_num in reversed(range(0,len(self.model),1)):
            self.dense["layer_"+str(lay_num)].weights=\
                np.loadtxt('Model/'+'layer_'+str(lay_num)+'_weights.csv', delimiter=',')
            
            self.dense["layer_"+str(lay_num)].biases=\
                np.loadtxt('Model/'+'layer_'+str(lay_num)+'_biases.csv',  delimiter=',')
                
    def test(self,x_test,y_test):
        self.load()
        
        self.eval(x_test,y_test)
        self.loss(y_test)

        predictions = np.argmax(self.result, axis=1)
        accuracy = np.mean(predictions==y_test)

       
        
        print(  f'acc: {accuracy:.3f}, ' +f'loss: {self.losss:.3f} (' )
        
        return  self.losss,accuracy,predictions