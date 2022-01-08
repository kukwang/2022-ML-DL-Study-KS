import numpy as np
from compute_utils import *
def Linear_Regression_GD(x,y,b,w,learning_rate=0.01,epochs=10000):
    '''
    설명 : gradient descent를 통해서 b,w를 epoch만큼 업데이트 시키고 return w,b 
    x : [bsz,representation]
    y : [bsz,]
    w : [representation]
    b : [1]
    '''
    
    return w,b

def Linear_Regression(x,y,b,w,learning_rate=0.01,epochs=1000,batch_size=16):
    '''
    Linear Regression based on SGD
    input
        x       : [bsz,representation]
        y       : [bsz,1]
    parameters
        w       : [rep,number of class]
        b       : [1,number of class]
    output
        w       : [rep,number of class]
        b       : [1,number of class]
    '''
   
    return w,b

def Logistic_Regression(x,y,w,b,number_of_class,learning_rate=0.0001,epochs=300000,batch_size = 32):
    '''
    Logistic Regression based on SGD
    input 
        x       : [data_size,rep]
        y       : [data_size,1]   
    parameters
        w       : [rep,number of class]
        b       : [1,number of class]
    output
        w       : [rep,number of class]
        b       : [1,number of class]
    '''
    y = one_hot(y, number_of_class)
    for i in range(epochs):
        batch_mask = np.random.choice(x.shape[0], batch_size)
        x_batch = x[batch_mask]
        y_batch = y[batch_mask]
        z = sigmoid(x_batch.dot(w)+b)
        dw = (y_batch-z).T.dot(x_batch)/batch_size
        db = (y_batch-z).mean(axis=0)
        w += learning_rate * dw.T
        b += learning_rate * db
    return w, b

def SVM(x,y,w,b,number_of_class,C=30,learning_rate=0.001,epochs=10000):
    '''
    이진 linear SVM만을 가정
    input
        x       : [data size,rep]
        y       : [data size,1]
    parameters
        w       : [rep,number of class]
        b       : [1, number of class]
    output
        w       : [rep,number of class]
        b       : [1, number of class]
    '''
    
    return w,b

def Naive_Bayes(x,y,number_of_class):
    '''
    input
        x : [data size, rep] : data_size x rep matrix
        y : [data size, 1] : data_size x 1 matrix
    output
        p_rep_status    : y의 상태에 따라 rep=1일 확률
        p_pos           : y=1인 데이터 발생 확률
    '''

    # 구간을 균일하게 나누고 그 구간을 저장
    bins = np.linspace(np.min(x, axis=0), np.max(x, axis=0), num=5)
    # input x를 discretize한 값을 저장할 변수 선언. [features, samples]
    discretize = np.zeros((x.shape[1], x.shape[0]))
    # x를 discretize, np.digitize가 다차원 연산이 안되서 for문으로 처리
    # np.digitize의 동작 방식때문에 x랑 bins에 transpose해서 넣음
    for i in range(x.shape[1]):
        discretize[i] = np.digitize(x.T[i], bins.T[i])
    # 헷갈리니 discretize를 transpose
    discretize = discretize.T

    num_of_label_cancer = collections.Counter(y)[1]
    # p(y=1)
    py_1 = num_of_label_cancer / y.size
    # p(y=0)
    py_0 = 1 - py_1
    # p(x|y=1)
    pxy_1 = np.prod()
    # p(x|y=0)
    pxy_0 =


    return x_threshold,p_rep_status,p_pos

def K_Means(data):
    '''
    input
        data : [data size, rep]
    '''
    number_of_centroids = 2
    return clusters, centroids


def Random_Forest():
    pass