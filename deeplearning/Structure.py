import  numpy                       as      np
import  utils
import  losses
import  nets
from    nets.Activation_functions   import  *
from    nets.Layers                 import  Dense,Dropout

def activation_function_dic(num:int):
    if      num ==  1:
        return ReLU()
    elif    num ==  2:
        return LeakyReLu()
    elif    num ==  3:
        return tangenthyperbolic()
    elif    num ==  4:
        return sigmoid()
    elif    num ==  5:
        return Softmax()
    elif    num ==  6:
        return Linear()

class structure():
    def __init__(self,) -> None:
        
        self.model=utils.model_Architecture
        
        # Empty Dictionary of layers
        self.dense  ={}
        self.act_Fn ={}
        self.dropout={}
        
        #loss
        if self.model[-1][-2]==4 or utils.task=='Regression':
            self.loss_activation=losses.MSE()
        else:
            self.loss_activation=losses.Loss_CategoricalCrossentropy()
        
        #self.optimizer
        if utils.optimizer_name=='SGD':
            self.optimizer = nets.SGD(learning_rate=utils.learning_rate, decay=utils.weight_decay,momentum=utils.opt_momentum)
            
        #input and making shells
        self.input=utils.img_channels*utils.img_height*utils.img_width
        for lay_num in range(0,len(self.model),1):
            self.dense["layer_"+str(lay_num)]=Dense(self.input,
                                                                self.model[lay_num][0],
                                                                self.model[lay_num][1],
                                                                self.model[lay_num][2],
                                                                self.model[lay_num][3],
                                                                self.model[lay_num][4])
            self.input=self.model[lay_num][0]

            self.act_Fn["layer_"+str(lay_num)]=activation_function_dic(self.model[lay_num][5])

            self.dropout["layer_"+str(lay_num)]=Dropout(self.model[lay_num][6])
        
    def forward(self,X,y):
        for lay_num in range(0,len(self.model),1):
            self.dense["layer_"+str(lay_num)].forward(X)
            X=self.dense["layer_"+str(lay_num)].output
            
            self.act_Fn["layer_"+str(lay_num)].forward(X)
            X=self.act_Fn["layer_"+str(lay_num)].output
            if not self.model[lay_num][6]==0:
                self.dropout["layer_"+str(lay_num)].forward(X)
                X=self.dropout["layer_"+str(lay_num)].output

        self.result=X
        
        #accuracy
        predictions = np.argmax(X, axis=1)
        self.accuracy = np.mean(predictions==y)
        
    def loss(self,y):
        
        
        self.Legularization_loss=0
        if not utils.L2==0 :
            for lay_num in range(0,len(self.model),1):
                self.Legularization_loss+=self.loss_activation.regularization_loss(self.dense["layer_"+str(lay_num)])
                
        self.losss = self.loss_activation.forward(self.result, y)+self.Legularization_loss
                
        self.loss_activation.backward(self.result, y)
        self.dval = self.loss_activation.dinputs
            
    def backward(self):
        for lay_num in reversed(range(0,len(self.model),1)):
            if not self.model[lay_num][6]==0:
                self.dropout["layer_"+str(lay_num)].backward(self.dval)
                self.dval=self.dropout["layer_"+str(lay_num)].dinputs
                
            self.act_Fn["layer_"+str(lay_num)].backward(self.dval)
            self.dval=self.act_Fn["layer_"+str(lay_num)].dinputs
            
            self.dense["layer_"+str(lay_num)].backward(self.dval)
            self.dval=self.dense["layer_"+str(lay_num)].dinputs

    def optimization(self,optimizer):

        optimizer.pre_decay()
        for lay_num in reversed(range(0,len(self.model),1)):
            optimizer.mid_update_params(self.dense["layer_"+str(lay_num)])
        optimizer.post_itter()
        
    def eval(self,X,y):
        for lay_num in range(0,len(self.model),1):
            self.dense["layer_"+str(lay_num)].forward(X)
            X=self.dense["layer_"+str(lay_num)].output
            
            self.act_Fn["layer_"+str(lay_num)].forward(X)
            X=self.act_Fn["layer_"+str(lay_num)].output
        self.result=X
        
        self.accuracy = np.mean( np.argmax(X, axis=1) ==y)
        
    def save(self):
        for lay_num in range(0,len(self.model),1):
            np.savetxt('Model/'+'layer_'+str(lay_num)+'_weights.csv',
                            self.dense["layer_"+str(lay_num)].weights,
                            delimiter=',')
            
            np.savetxt('Model/'+'layer_'+str(lay_num)+'_biases.csv',
                            self.dense["layer_"+str(lay_num)].biases,
                            delimiter=',')