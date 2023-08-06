import numpy as np
# import tqdm
from .Structure import structure
import  utils


class train(structure):
    def train(self,         
                datas,labels,
                ifeval,
                train_net,
                first,last,validation,
                x_test,y_test,ifTEST):
        epoch=utils.num_epochs
        batch_siz=utils.batch_size
        
        lo_list=[]
        accu_list=[]

        lo_list_test=[]
        accu_list_test=[]
        accu_list_validation=[]
        lo_list_validation=[]
        for epoch in range(epoch):
            loss_list=[]
            accuracy_list=[]
            for i in reversed(range(batch_siz)):
                #Eval
                if i==0 and ifTEST==True:

                    self.eval(x_test,y_test)
                    self.loss(y_test)
                    y_pre_test=np.argmax(np.round(self.result,10),axis=1)
                                
                    accu_list_test.append(self.accuracy)
                    lo_list_test.append(self.losss)
                if i==0 and ifeval==True:
                    
                    x_validation=datas[validation]
                    y_validation=labels[validation]
                    
                    self.eval(x_validation,y_validation)
                    self.loss(y_validation)
                    y_pre_valid=np.argmax(np.round(self.result,10),axis=1)
                                
                    accu_list_validation.append(self.accuracy)
                    lo_list_validation.append(self.losss)


                if i==batch_siz-1:
                    X=datas[last]
                    y=labels[last]
                else:
                    X=datas[first[i,:]]
                    y=labels[first[i,:]]

                self.forward(X,y)
                self.loss(y)

                predictions = np.argmax(np.round(self.result,10),axis=1)
                accuracy = np.mean(predictions==y)
                loss_list.append(self.losss)
                accuracy_list.append(accuracy)
                if train_net==True:
                    self.backward()
                    self.optimization(self.optimizer)

            lo_list     .append(np.mean(np.array(loss_list)))
            accu_list   .append(np.mean(np.array(accuracy_list)))
            
            
            print(f'epoch: {epoch}, ' +
                    f'acc: {np.mean(np.array(accuracy_list)):.3f}, ' +
                    f'loss: {np.mean(np.array(loss_list)):.3f} ' +
                    f'acc_validation: {accu_list_validation[epoch]:.3f}, ' +
                    f'loss_validation: {lo_list_validation[epoch]:.3f} ' +
                    f'lr: {self.optimizer.current_learning_rate}')
            if ifTEST==True:            
                print(f'acc_test: {accu_list_test[epoch]:.3f}, ' +
                    f'loss_test: {lo_list_test[epoch]:.3f} ' )
        self.save()
        return  lo_list,accu_list,lo_list_validation,accu_list_validation,lo_list_test,accu_list_test,predictions,y_pre_valid,y_pre_test , y,y_validation,y_test

def train_sto(net,         
            datas,labels,
            first,last,validation):
    epoch=utils.num_epochs
    batch_siz=utils.batch_size

    loss_list=[]
    accuracy_list=[]

    lo_list_test=[]
    accu_list_test=[]

    numnum=first.shape[0]*first.shape[1]+last.shape[0]
    itter=0
    ####################################################
    mini_X=[]
    mini_y=[]
    for i in range(32):
                if i==32-1:
                    X=datas[last]
                    y=labels[last]
                    
                    mini_X.extend(X)
                    mini_y.extend(y)
                else:
                    X=datas[first[i,:]]
                    y=labels[first[i,:]]

                    mini_X.extend(X)
                    mini_y.extend(y)
    X=np.array(mini_X)
    y=np.array(mini_y).reshape([-1,1])
    #####################################################
    for epoch in range(epoch):

        for i in range(len(y)):
            #Eval################################################   
            x_test=datas[validation]
            y_test=labels[validation]
            
            net.eval(x_test,y_test)
            net.loss(y_test)
            y_pre_valid=np.argmax(np.round(net.result,10),axis=1)
                        
            accu_list_test.append(net.accuracy)
            lo_list_test.append(net.losss)
            #######################################################
            Xin=X[i].reshape([1,3072])
            net.forward(Xin,y[i])
            net.loss(y[i])

            predictions = np.argmax(np.round(net.result,10),axis=1)
            accuracy = np.mean(predictions==y[i])
            loss_list.append(net.losss)
            accuracy_list.append(accuracy)
            net.backward()
            net.optimization(net.optimizer)
            if i%100==0:
                print(f'epoch,data: {epoch,i}, ' +
                        f'acc: {accuracy_list[i]:.3f}, ' +
                        f'loss: {loss_list[i]:.3f} ' +
                        f'acc_vali: {accu_list_test[i]:.3f}, ' +
                        f'loss_vali: {lo_list_test[i]:.3f} ' +
                        f'data_loss: {net.losss:.3f}, ' +
                        # f'reg_loss: {net.regularization_loss:.3f}), ' +
                        f'lr: {net.optimizer.current_learning_rate}')
        itter+=1
        break
    return loss_list,accuracy_list,lo_list_test,accu_list_test,predictions,y_pre_valid,y,y_test

def train_GD(net,         
            datas,labels,
            first,last,validation):
    epoch=utils.num_epochs
    batch_siz=utils.batch_size

    loss_list=[]
    accuracy_list=[]

    lo_list_test=[]
    accu_list_test=[]

    numnum=first.shape[0]*first.shape[1]+last.shape[0]
    itter=0
    ####################################################
    mini_X=[]
    mini_y=[]
    for i in range(32):
                if i==32-1:
                    X=datas[last]
                    y=labels[last]
                    
                    mini_X.extend(X)
                    mini_y.extend(y)
                else:
                    X=datas[first[i,:]]
                    y=labels[first[i,:]]

                    mini_X.extend(X)
                    mini_y.extend(y)
    X=np.array(mini_X)
    y=np.array(mini_y)
    
    x_test=datas[validation]
    y_test=labels[validation]
    #####################################################
    for epoch in range(epoch):

        #Eval################################################   
        net.eval(x_test,y_test)
        net.loss(y_test)
        y_pre_valid=np.argmax(np.round(net.result,10),axis=1)
                    
        accu_list_test.append(net.accuracy)
        lo_list_test.append(net.losss)
        #######################################################
        net.forward(X,y)
        net.loss(y)

        predictions = np.argmax(np.round(net.result,10),axis=1)
        accuracy = np.mean(predictions==y)
        loss_list.append(net.losss)
        accuracy_list.append(accuracy)
        net.backward()
        net.optimization(net.optimizer)

        print(f'epoch: {epoch}, ' +
                f'acc: {accuracy_list[epoch]:.3f}, ' +
                f'loss: {loss_list[epoch]:.3f} ' +
                f'acc_vali: {accu_list_test[epoch]:.3f}, ' +
                f'loss_vali: {lo_list_test[epoch]:.3f} ' +
                f'data_loss: {net.losss:.3f}, ' +
                # f'reg_loss: {net.regularization_loss:.3f}), ' +
                f'lr: {net.optimizer.current_learning_rate}')    
    return loss_list,accuracy_list,lo_list_test,accu_list_test,predictions,y_pre_valid,y,y_test