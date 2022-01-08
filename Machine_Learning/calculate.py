from compute_utils import *
from arguments import get_arguments
from methods import *
from sklearn.datasets import load_iris, make_blobs, load_breast_cancer
import argparse
import numpy
import collections

def main():
    # argument parsing
    args = argparse.ArgumentParser()
    args = get_arguments()

    if args.methods=='Linear_Regression':
        # y = 2x+5
        x = np.array([2,6,9])
        y = np.array([9,17,23])
        w = np.random.uniform(-1,1,1)
        b = np.random.uniform(-1,1,1)
        w,b = globals()[args.methods](x,y,w,b)
        '''
        PREDICTION
        '''

    elif args.methods=='Logistic_Regression':
        x, y = load_iris(return_X_y=True)
        number_of_class = len(collections.Counter(y))
        x_train,y_train,x_test,y_test = divide_dataset(x,y)
        w = np.random.uniform(-1,1,(x.shape[1],number_of_class))
        b = np.random.uniform(-1,1,(number_of_class))
        w, b = globals()[args.methods](x_train,y_train,w,b,number_of_class)
        '''
        PREDICTION
        '''
        y_test_oh = one_hot(y_test, number_of_class)
        pred = softmax(x_test.dot(w) + b)
        print("accuracy: ", np.round((pred.argmax(1) == y_test_oh.argmax(1)).mean(), 4))

    elif args.methods=='SVM':
        x,y = make_blobs(n_samples=150,centers=2,random_state=20)
        y = np.sign(y-0.5).astype(np.int64)
        number_of_class = len(collections.Counter(y))
        x_train,y_train,x_test,y_test = divide_dataset(x,y)
        w = np.ones((x.shape[1]))
        b = np.ones((1))
        w,b = globals()[args.methods](x_train,y_train,w,b,number_of_class)

        '''
        PREDICTION
        '''

    elif args.methods=='Naive_Bayes':
        data = load_breast_cancer()
        x = data['data']
        y = data['target']
        x_train,y_train,x_test,y_test = divide_dataset(x,y)
        number_of_class = len(collections.Counter(y))
        x_threshold,p_rep_status,p_pos = globals()[args.methods](x_train,y_train,number_of_class)

        '''
        PREDICTION
        '''
      
    elif args.methods =='K_Means':
        x1 = np.random.uniform(-5,0,100)
        y1 = np.random.uniform(-5,0,100)
        x2 = np.random.uniform(5,10,50)
        y2 = np.random.uniform(5,10,50)
        data1 = np.vstack([x1,y1]).T
        data2 = np.vstack([x2,y2]).T
        data = np.concatenate([data1,data2],axis=0)
        clusters,centroids =globals()[args.methods](data)
        '''
        PREDICTION
        '''

if __name__=='__main__':
    main()