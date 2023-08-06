import numpy as np
import matplotlib.pyplot as plt

def result_plot(model_name:str,
             plot_desc:str,
             data_1:list,
             data_2:list,
             DPI=100,
             axis_label_size=15,
             x_grid=0.1,
             y_grid=0.5,
             axis_size=12):
    
    assert(len(data_1)==len(data_2))
    
    '''
        in this function by getting the two list of data result will be ploted as the 
        TA desired
        
        Parameters
        ----------
        model_name:str : dosent do any thing in this vesion but can implemented to save image file
            with the given name
        
        plot_desc:str : define the identity of the data i.e. Accuracy , loss ,...
        
        data_1:list     : list of first data set .
        data_2:list     : list of second data set .
        
        optional
        
        DPI=100             :   define the quality of the plot
        axis_label_size=15  :   define the label size of the axises
        x_grid=0.1          :   x_grid capacity
        y_grid=0.5          :   y_grid capacity
        axis_size=12        :   axis number's size
        
        
        Returns
        -------
        none      : by now 
        

        See Also
        --------
        size of the two list must be same 

        Notes
        -----
        size of the two list must be same 

        Examples
        --------
        >>> result_plot("perceptron",
             "loss",
             data_1,
             data_2)

        '''
    
    fig , ax = plt.subplots(1,figsize=(10,5) , dpi=DPI)

    fig.suptitle(f"Train and validation "+plot_desc,y=0.95 , fontsize=20)

    epochs = range(len(data_1))
    ax.plot(epochs, data_1, 'b',linewidth=3, label='tarin '+ plot_desc)
    ax.plot(epochs, data_2, 'r',linewidth=3, label='validation '+plot_desc)

    ax.set_xlabel("epoch"       ,fontsize=axis_label_size)
    ax.set_ylabel(plot_desc     ,fontsize=axis_label_size)
    if x_grid:
        ax.grid(axis="x",alpha=0.1)
    if y_grid:
        ax.grid(axis="y",alpha=0.5)
    

    ax.legend(loc=0,prop={"size":9})

    ax.tick_params(axis="x",labelsize=axis_size)
    ax.tick_params(axis="y",labelsize=axis_size)

    #spine are borde line of plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # ax.set_ylim([8.48,8.6])
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()
    
# https://www.binarystudy.com/2021/09/how-to-load-preprocess-visualize-CIFAR-10-and-CIFAR-100.html
def random_plotter( datas,
                    labels,
                    meta,
                    name=False,
                    figure_size=28,
                    DPI=200,
                    row=10,column=10,
                    pic_in_class=5000
                    ):
    
    plt.figure(figsize=(figure_size, figure_size), dpi=DPI)

    for i in range(row):
        for j in range(column):
            plt.subplot(row, column, i+1 + j*10)
            ranff=np.random.randint(0,pic_in_class,1)
            id=np.array(np.where(labels == j)).reshape(pic_in_class)[ranff]
            test=datas[id].reshape(3, 32, 32).transpose(1, 2, 0)

            plt.title(meta[int(labels[id])])
            plt.imshow(test,aspect="auto")
    
    plt.tight_layout()
    if name:
        plt.savefig(name+'.png', bbox_inches='tight')
    plt.show()

# https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix
def compute_confusion_matrix(true, pred):
  '''Computes a confusion matrix using numpy for two np.arrays
  true and pred.

  Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

  However, this function avoids the dependency on sklearn.'''

  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))

  for i in range(len(true)):
    result[true[i]][pred[i]] += 1

  return result

# https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix
def plot_confusion_matrix(model_name,df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.colorbar()
    plt.title(title)
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()

def result_plot2(model_name:str,
             plot_desc:str,
             data_1:list,
             data_2:list,
             data_3:list,
             DPI=100,
             axis_label_size=15,
             x_grid=0.1,
             y_grid=0.5,
             axis_size=12):
    
    assert(len(data_1)==len(data_2))

    
    fig , ax = plt.subplots(1,figsize=(10,5) , dpi=DPI)

    fig.suptitle(f"Train , validation and Test "+plot_desc,y=0.95 , fontsize=20)

    epochs = range(len(data_1))
    ax.plot(epochs, data_1, 'b',linewidth=3, label=' tarin '+ plot_desc)
    ax.plot(epochs, data_2, 'r',linewidth=3, label=' validation '+plot_desc)
    ax.plot(epochs, data_3, 'g',linewidth=3, label=' test '+plot_desc)

    ax.set_xlabel("epoch"       ,fontsize=axis_label_size)
    ax.set_ylabel(plot_desc     ,fontsize=axis_label_size)
    if x_grid:
        ax.grid(axis="x",alpha=0.1)
    if y_grid:
        ax.grid(axis="y",alpha=0.5)
    

    ax.legend(loc=0,prop={"size":9})

    ax.tick_params(axis="x",labelsize=axis_size)
    ax.tick_params(axis="y",labelsize=axis_size)

    #spine are borde line of plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # ax.set_ylim([0.0,0.25])
    plt.savefig(model_name+'.png', bbox_inches='tight')
    plt.show()
