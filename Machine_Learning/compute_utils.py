import numpy as np
import collections


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    '''
    input  : [bsz,c]
    output : [bsz,c]
    '''
    max_a = a.max()
    a -= max_a
    exp_a = np.exp(a)
    if a.ndim == 1:
        sum_a = np.sum(exp_a)
        return exp_a/sum_a
    else:
        sum_a = np.sum(exp_a, axis=1).reshape(np.sum(exp_a, axis=1).shape[0], 1)
        return exp_a / sum_a


def one_hot(y, k):
    '''
    input 
        y : [bsz,1] : batch_size x 1
        k : [1,] : 1
    
    output
        y_oh : [bsz,k] : batch_size x k
    '''
    batch_size = y.shape[0]
    y_one_hot = np.zeros((batch_size, k))  # create bsz x k matrix filled with zeros
    for i in range(batch_size):
        y_one_hot[i][y[i]] = 1  # write 1 in the corresponding index in every row

    return y_one_hot


def cross_entropy_loss(y, t):
    '''
    input
        y : [bsz, label]           label들은 one-hot 형태인지 아닌지는 모름
        t : [bsz, # of class]

    output
        loss : [bsz, 1]
    '''
    batch_size = y.shape[0]
    loss = np.zeros((batch_size, 1))
    for i in range(batch_size):
        loss[i][0] = -np.dot(t[i], np.log(y[i]))

    return loss


def divide_dataset(x, y):
    train_mask = np.random.choice(x.shape[0], int(x.shape[0] * 0.8), replace=False)
    test_mask = np.delete(np.arange(len(x)), train_mask)
    x_train = x[train_mask]
    y_train = y[train_mask]
    x_test = x[test_mask]
    y_test = y[test_mask]
    return x_train, y_train, x_test, y_test


def discretize(x, bound):
    discretize = np.zeros((x.shape[1], x.shape[0]))
    # x를 discretize, np.digitize가 1차원만 연산이 가능하기에 for문으로 처리
    # np.digitize의 동작 방식때문에 x랑 sep을 transpose한 뒤에 각각의 row를 넣음
    for i in range(x.shape[1]):
        discretize[i] = np.digitize(x.T[i], bound.T[i])
    # element들을 {1, 2} -> {0, 1}로 바꾸고 transpose -> [data_size, rep]
    discretize = (discretize-1).T
    return discretize