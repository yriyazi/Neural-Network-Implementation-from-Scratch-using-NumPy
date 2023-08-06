#%%
import  glob
import  os
import  nets
import  losses
import  utils
import  dataloaders
import  deeplearning

import  matplotlib.pyplot   as  plt
import  numpy               as  np
from    pathlib             import  Path
np.random.seed(utils.seed)


#%%
#reading data
path = Path("datasets")
datas,labels    =dataloaders.cifar10_reader(path,"datasets/data_*","/Question1/")
x_test,y_test   =dataloaders.cifar10_reader(path,"datasets/test_*","/Question1/")
# pre process
data=dataloaders.data_pre_pro(datas,x_test)
datas =data.datas
x_test=data.x_test

#%%
test=deeplearning.test()
#%%
test_loss,test_accuracy,predictions =test.test(x_test,y_test)

#%%
cof=utils.compute_confusion_matrix(y_test, predictions)
utils.plot_confusion_matrix("test",cof)