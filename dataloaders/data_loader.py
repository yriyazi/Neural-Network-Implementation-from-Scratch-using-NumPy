import  glob
import  os
import  utils
import  numpy   as      np
from    pathlib import  Path

#use default univecitiy of washington help
def unpickle(file):
    '''
        loading the binary file and reading the data
    '''
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
# class stratify():
#     def __init__(self,
#                  index,
#                  train_valid_split=7/10,
#                  ) -> None:
        
#         self.each_class =5000
#         self.stop       =90000
#         self.classes    =utils.num_classes
#         self.batch_size =utils.batch_size
#         self.All_batch  =self.batch_size/self.train_valid_split
#         self.index      =index

#         if      utils.batch_size==1:
#             self.mini()
#         elif    utils.batch_size==utils.trai_size:
#             self.Batch()
#         else:
#             self.stratify()

#     def Batch(self,):
#         pass

    
#     def stratify(self,):
#         '''
#         dividing the data to equall batches
        
#         return
#         -------
#         Tempp, Temp , validation: m-n-1 batch of data , (m+n) th batch , (m-n) stacked validation batch
        
#         see also
#         -------
#         because the batch size woudnt be same for all batches and coudnt add it to same list i return it 
#         sepratly.
        
#         example
#         -------
#         for i in range(10):
#             print(np.array(np.where(labels[first[19,:]] == i)).shape)
        
#         >>>
#         '''
#         batch_len=int(np.round((len(self.index)/self.classes)/(self.All_batch-1)))
        
#         for i in range(self.All_batch):
            
#             if i > self.All_batch - self.batch_size-1:
#                 if i == self.All_batch-1:
#                     for j in range(self.classes):
#                         if j == 0 :
#                             Temp=np.array(np.where(self.index == j)).reshape(self.each_class)[0+batch_len*i:]
#                         else:
#                             Temp=np.hstack([Temp,np.array(np.where(self.index == j)).reshape(self.each_class)[0+batch_len*i:]])
                        
#                     return Tempp,Temp,Validation
                
#                 else:
#                     for j in range(self.classes):
#                         if j == 0 :
#                             Temp=np.array(np.where(self.index == j)).reshape(self.each_class)[0+batch_len*i:batch_len*(i+1)]
#                         else:
#                             Temp=np.hstack([Temp,np.array(np.where(self.index == j)).reshape(self.each_class)[0+batch_len*i:batch_len*(i+1)]])
#                 if i==self.All_batch - self.batch_size:
#                     Tempp=Temp
#                 else:
#                     Tempp=np.vstack([Tempp,Temp])            
#                 if i > self.stop-(self.All_batch - self.batch_size-1):
#                     return Tempp
#             else:
#                 for j in range(self.classes):
#                         if j+i == 0 :
#                             Validation=np.array(np.where(self.index == j)).reshape(self.each_class)[0+batch_len*i:batch_len*(i+1)]
#                         else:
#                             Validation=np.hstack([Validation,np.array(np.where(self.index == j)).reshape(self.each_class)[0+batch_len*i:batch_len*(i+1)]])
   
def stratify(All_batch  :int,
             batch_size :int,
             index      :np.array,
             classes    :int,
             stop       =10000,
             each_class =5000):
    '''
    dividing the data to equall batches
    
    return
    -------
    Tempp, Temp , validation: m-n-1 batch of data , (m+n) th batch , (m-n) stacked validation batch
    
    see also
    -------
    because the batch size woudnt be same for all batches and coudnt add it to same list i return it 
    sepratly.
    
    example
    -------
    for i in range(10):
        print(np.array(np.where(labels[first[19,:]] == i)).shape)
    
    >>>
    '''
    batch_len=int(np.round((len(index)/classes)/(All_batch-1)))
    
    for i in range(All_batch):
        
        if i > All_batch - batch_size-1:
            if i == All_batch-1:
                for j in range(classes):
                    if j == 0 :
                        Temp=np.array(np.where(index == j)).reshape(each_class)[0+batch_len*i:]
                    else:
                        Temp=np.hstack([Temp,np.array(np.where(index == j)).reshape(each_class)[0+batch_len*i:]])
                    
                return Tempp,Temp,Validation
            
            else:
                for j in range(classes):
                    if j == 0 :
                        Temp=np.array(np.where(index == j)).reshape(each_class)[0+batch_len*i:batch_len*(i+1)]
                    else:
                        Temp=np.hstack([Temp,np.array(np.where(index == j)).reshape(each_class)[0+batch_len*i:batch_len*(i+1)]])
            if i==All_batch - batch_size:
                Tempp=Temp
            else:
                Tempp=np.vstack([Tempp,Temp])            
            if i > stop-(All_batch - batch_size-1):
                return Tempp
        else:
            for j in range(classes):
                    if j+i == 0 :
                        Validation=np.array(np.where(index == j)).reshape(each_class)[0+batch_len*i:batch_len*(i+1)]
                    else:
                        Validation=np.hstack([Validation,np.array(np.where(index == j)).reshape(each_class)[0+batch_len*i:batch_len*(i+1)]])
  
def cifar10_reader(path,
                   reading_type="datasets/data_*",
                   paret_folder_name="/Question1/",
                   ):
     
    parent=os.path.dirname(path.parent.absolute())
    ii=1

    for i in glob.glob(reading_type):
        x=parent+(paret_folder_name+i[:])
        Temp=unpickle(x)
        if ii == 1 :
            labels=np.array(Temp[b'labels'])
            datas =np.array(Temp[b'data'])
        else :
            labels=np.hstack([labels,np.array(Temp[b'labels'])])
            datas =np.vstack([datas ,np.array(Temp[b'data']  )])
        ii=ii+1

    return datas,labels

def cifar10_meta(path,
                   reading_type="datasets/data_*",
                   paret_folder_name="/Question1/",
                   ):
    parent=os.path.dirname(path.parent.absolute())
    ii=1

    for i in glob.glob(reading_type):
        x=parent+(paret_folder_name+i[:])
        Temp=unpickle(x)
    
    meta=Temp[b'label_names']
    meta=[str(meta[i])[2:-1] for i in range(len(meta))]
        
    return meta

class data_pre_pro():
    def __init__(self,datas,x_test) -> None:
        self.datas =datas
        self.x_test=x_test
        
        if   utils.pre_pro=='standardize':
            self.standardize()
        elif utils.pre_pro=='Normalization':
            self.Normalization()
        else:
            self.nothing()

    def Normalization(self):
        self.min=np.min(self.datas)
        self.max=np.max(self.datas)
        
        self.datas =(self.datas-self.min) /(self.max-self.min)
        self.x_test=(self.x_test-self.min)/(self.max-self.min)
        

    def standardize(self,):
        self.mean=np.mean(self.datas)
        self.std =np.std(self.datas)
        
        self.datas =(self.datas-self.mean) /self.std
        self.x_test=(self.x_test-self.mean)/self.std
    def nothing(self,):
            pass