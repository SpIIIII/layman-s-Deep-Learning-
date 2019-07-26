import numpy as np
import pandas as pd
import sys


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y
        return self

    def calc_nearestest(self,labels):
        A = np.array(labels)
        is_more = np.bincount(A)

        if (any(is_more[is_more>1])):
            S = pd.DataFrame({'col 1':A,'col 2':np.arange(len(A))})
            gb_S = S.groupby(S['col 1'])
            largest = gb_S.size().nlargest()
            lrg_val = largest[largest==largest.max()].index.values
            if len(lrg_val)==1:
                return A[largest.loc[lrg_val[0]]]
            else:
                x=[np.mean(S[S['col 1']==i].index.values) for i in lrg_val]
                return A[np.where(np.argmin(x))][0]
            
        else: 
            return(A[0])



    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        
        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), qunp.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                #print(X[i_test],self.train_X[i_train])
                ranges = X[i_test]-self.train_X[i_train]
                dists[i_test,i_train] =  np.sum(np.abs(ranges))
                # TODO: Fill dists[i_test][i_train]
                pass
        return dists
    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = np.sum(np.abs(self.train_X   - X[i_test]),axis=1)
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
            pass
        return dists
    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
        dists = np.zeros((num_test, num_train), np.int16)
        # print('test',X[:,None].shape,'train',self.train_X.shape)
        inter0 = X[:,None].astype(np.int16)-self.train_X.astype(np.int16)
        # print('inter0 shape',inter0.shape())
        inter1 = np.abs(inter0)
        dists = np.sum(inter1,axis = 2)
        
        # TODO: Implement computing all distances with no loops!
        pass
        return dists

    def predict_labels_binary(self, dists):
        '''
        Returns model predictions for binary classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        # print(self.train_y.shape)
        # print(dists.shape)
        
        for i in range(num_test):
            k_smallest = np.argpartition(dists[i],self.k)
            pred[i] =  np.mean(self.train_y[k_smallest[:3]])>0.5
            
            
        return pred

    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        
        pred = np.zeros(num_test, np.int)
        # print(self.train_X.shape,self.train_y.shape    )
        for i in range(num_test):
            k_smallest = dists[i].argsort()[:self.k]
            pred[i] = self.train_y[self.calc_nearestest(k_smallest)]
    
            pass
        
        return pred

    
    def pp (self):
        return 