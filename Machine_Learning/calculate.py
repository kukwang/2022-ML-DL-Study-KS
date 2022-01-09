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
        # x_test를 먼저 discretize (구간에 따라 0, 1)
        x_test = discretize(x_test, x_threshold)
        # 모든 data의 모든 rep에 대한 p(y=0|x)과 p(y=0|x)을 저장할 matrix 선언
        py0x_tmp = np.zeros((x_test.shape[0], x_test.shape[1]))
        py1x_tmp = np.zeros((x_test.shape[0], x_test.shape[1]))
        # 모든 data에 대해
        for data in range(x_test.shape[0]):
            # 모든 rep에 대해 rep == 0: 1-p(x=1|y=0), rep == 1: p(x=1|y=0) 저장
            for rep in range(x_test.shape[1]):
                if x_test[data][rep] == 0:
                    py0x_tmp[data][rep] = 1-p_rep_status[0][rep]
                else:
                    py0x_tmp[data][rep] = p_rep_status[0][rep]
            # 모든 rep에 대해 rep == 0: 1-p(x=1|y=1), rep == 1: p(x=1|y=1) 저장
            for rep in range(p_rep_status.shape[1]):
                # 모든 rep에 대해 rep가 0이면 1-학습한 확률, 1이면 학습한 확률을 해당 위치에 저장
                if x_test[data][rep] == 0:
                    py1x_tmp[data][rep] = 1-p_rep_status[1][rep]
                else:
                    py1x_tmp[data][rep] = p_rep_status[1][rep]
        # 각 rep의 확률들을 곱해서 p(y=0|x)과 p(y=1|x) 계산 : 둘 다 [data_size, ]
        py0x = np.prod(py0x_tmp, axis=1)
        py1x = np.prod(py1x_tmp, axis=1)

        # 구한 p(y|x) 중 큰 값을 예측값으로 결정 : [data_size, ]
        pred = np.zeros((x_test.shape[0]))
        for data in range(x_test.shape[0]):
            if py0x[data] > py1x[data]:
                pred[data] = 0
            else:
                pred[data] = 1

        print("pred: ", pred)
        print("y_test", y_test)
        print(np.multiply(y_test, pred))
        print(np.multiply(y_test, 1-pred))
        print(np.multiply(1-y_test, pred))
        print(np.multiply(1-y_test, 1-pred))
        # precision과 recall 계산 후 출력
        TP = (np.multiply(y_test, pred)).sum()
        FN = (np.multiply(y_test, 1 - pred)).sum()
        FP = (np.multiply(1 - y_test, pred)).sum()
        TN = (np.multiply(1 - y_test, 1 - pred)).sum()

        print("TP: ", TP)
        print("FN: ", FN)
        print("FP: ", FP)
        print("TN: ", TN)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        accuracy = (TP + TN) / (TP + FN + FP + TN)
        print("precision: ", precision)
        print("recall: ", recall)
        print("accuracy: ", accuracy)
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