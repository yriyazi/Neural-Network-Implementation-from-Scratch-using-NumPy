#%%
import  glob
import  os
import  nets
import  losses
import  utils
import  dataloaders
import  deeplearning
import  tqdm
import  matplotlib.pyplot   as  plt
import  numpy               as  np
from    pathlib             import  Path

np.random.seed(utils.seed)
#reading data
path = Path("datasets")
datas,labels    =dataloaders.cifar10_reader(path,"datasets/data_*","/Question1/")
x_test,y_test   =dataloaders.cifar10_reader(path,"datasets/test_*","/Question1/")
# pre process
data=dataloaders.data_pre_pro(datas,x_test)
datas =data.datas
x_test=data.x_test
#%%
#statify
first,last,validation=dataloaders.stratify(round(utils.batch_size*1.1),utils.batch_size,labels,10)
# class names dictionary
meta=dataloaders.cifar10_meta(path,"datasets/*.meta","/Question1/")

net=deeplearning.train_loop.train()
#%%
from deeplearning.train_loop import train_sto

lo_list,accu_list,lo_list_test,accu_list_test,predictions,y_pre_valid,y,y_test=train_sto(net,datas,labels,
                                                                                    first,last,validation)
#%%
Name_ID=str(utils.model_Architecture[0][0])+"-"+str(utils.learning_rate)+"-"+str(utils.num_epochs)

utils.plot.result_plot("accuracy-"+Name_ID,"Accuracy",accu_list,accu_list_test,DPI=400)

utils.plot.result_plot("loss-"+Name_ID,"loss",lo_list,lo_list_test,DPI=400)


cof=utils.compute_confusion_matrix(y, predictions)
utils.plot_confusion_matrix("confussion matrix train-"+Name_ID,cof)

cof=utils.compute_confusion_matrix(y_test,y_pre_valid)
utils.plot_confusion_matrix("confussion matrix validation-"+Name_ID,cof)