#%%
import  glob
import  os
import  nets
import  losses
import  utils
import  dataloaders
import  deeplearning

import  numpy   as      np
from    pathlib import  Path
import matplotlib.pyplot as plt
np.random.seed(43)

#%%
#reading data
path = Path("datasets")
datas,labels    =dataloaders.cifar10_reader(path,"datasets/data_*","/Question1/")
x_test,y_test   =dataloaders.cifar10_reader(path,"datasets/test_*","/Question1/")
# pre process (normalize ...)
data=dataloaders.data_pre_pro(datas,x_test)
datas =data.datas
x_test=data.x_test
#statify
first,last,validation=dataloaders.stratify(38,32,labels,10)
# class names dictionary
meta=dataloaders.cifar10_meta(path,"datasets/*.meta","/Question1/")

# plotting classe randomly
# utils.plot.random_plotter(datas,labels,meta,name="random plot",row=10,column=10,DPI=100)
#%%
net=deeplearning.train_loop.train()
#%%
lo_list,accu_list,lo_list_validation,accu_list_validation,lo_list_test,accu_list_test,predictions,y_pre_valid,y_pre_test , y,y_validation,y_test=net.train(datas,labels,
                                                                                    True,
                                                                                    True,
                                                                                    first,last,validation,
                                                                                    x_test,y_test,True)



#%%
Name_ID=str(utils.model_Architecture[0][0])+"-"+str(utils.learning_rate)+"-"+str(utils.num_epochs)

# utils.plot.result_plot("accuracy-"+Name_ID,"Accuracy",accu_list,accu_list_test,DPI=400)
# utils.plot.result_plot("loss-"+Name_ID,"loss",lo_list,lo_list_test,DPI=400)
utils.plot.result_plot2("accuracy-"+Name_ID,"Accuracy",accu_list,accu_list_validation,accu_list_test,DPI=400)
utils.plot.result_plot2("loss-"+Name_ID    ,"loss"    ,lo_list  ,lo_list_validation  ,lo_list_test,DPI=400)

cof=utils.compute_confusion_matrix(y, predictions)
utils.plot_confusion_matrix("confusion matrix of train data"+Name_ID,cof,'confusion matrix of train data')

cof=utils.compute_confusion_matrix(y_validation, y_pre_valid)
utils.plot_confusion_matrix("confusion matrix of validation data"+Name_ID,cof,'confusion matrix of validation data')

cof=utils.compute_confusion_matrix(y_test, y_pre_test)
utils.plot_confusion_matrix("confusion matrix of test data"+Name_ID,cof,'confusion matrix of test data')