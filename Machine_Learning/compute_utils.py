import numpy as np
import collections
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    '''
    input  : [bsz,c]
    output : [bsz,c]
    '''

def one_hot(y, k):
    '''
    input 
        y : [bsz,1]
        k : [1,]
    
    output
        y_oh : [bsz,k]
    '''
  


def cross_entropy_loss(y,t):
    '''
    input
        y : [bsz, label]           label들은 one-hot 형태인지 아닌지는 모름
        t : [bsz, # of class]
    '''
   

def divide_dataset(x,y):
    train_mask = np.random.choice(x.shape[0],int(x.shape[0]*0.8),replace=False)
    test_mask = np.delete(np.arange(len(x)),train_mask)
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]
    return x_train,y_train,x_test,y_test