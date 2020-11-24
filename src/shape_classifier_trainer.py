#!/usr/bin/env python3

import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Local imports
from shape_classifier import shape_classifier
import constants

class shape_classifier_trainer:

    def __init__(self):
        #self.data_dir = constants.ML_DATA_DIR
        self.data_dir = './images/ml_samples/'
        #print(os.listdir(self.data_dir))
        self.sc = shape_classifier()
        self.X = None
        self.Y = None


    def load_data(self, start=0, stop=5527):
        classes = ['sphere', 'cuboid']
        C = 2
        N = C*(stop-start)
        D = 30*30
        XY = np.zeros((N,D+1))
        for i in range(C):
            c = classes[i]
            for j in range(start,stop):
                imgstr = self.data_dir + c + '_' + str(j) + '.png'
                img = cv2.imread(imgstr, cv2.IMREAD_GRAYSCALE)
                idx = i*(stop-start)+j
                XY[idx, :-1] = np.reshape(img, D)
                XY[idx, -1] = i
        np.random.shuffle(XY)
        self.X = XY[:,:-1]
        self.Y = XY[:,-1]
        # normalize
        # self.X /= 255.0
        # Xmean = np.mean(self.X, axis=0)
        # self.X -= Xmean
    
    
    def pca_plot(self):
        pca = PCA()
        pca.fit(self.X)
        labels = ['sphere', 'cuboid']
        igen = pca.components_
        for i in range(2):
            proj = np.matmul(self.X[self.Y==i,:], igen.transpose())
            plt.scatter(proj[:,0], proj[:,1], label=labels[i])
        plt.legend()
        plt.show()

        var = pca.explained_variance_ratio_
        cumvar = [var[0]]
        for i in range(1,len(var)):
            cumvar.append(cumvar[i-1]+var[i])
        plt.plot(range(1,len(var)+1), cumvar)
        plt.show()


    def pca_reduction(self, variance_thresh=0.95):
        pca = PCA()
        pca.fit(self.X)
        variances = pca.explained_variance_ratio_
        igen = pca.components_
        total_var = 0
        cutoff = -1
        for i in range(len(variances)):
            total_var += variances[i]
            if total_var >= variance_thresh:
                cutoff = i+1
                break
        self.X = np.matmul(self.X, igen[:cutoff,:].transpose())
        self.sc.eigenvectors = igen[:cutoff,:]
        print('Reduced to {} dimensions.'.format(cutoff))


    def train_model(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = self.X.shape[0]
        self.sc.fit(self.X[start:stop, :], self.Y[start:stop])


    def test_model(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = self.X.shape[0]
        predictions = self.sc.predict(self.X[start:stop,:])
        cm = confusion_matrix(y_true=self.Y[start:stop], y_pred=predictions)
        acc = np.sum(np.diag(cm)) / np.sum(cm) * 100
        print('Accuracy: {}%'.format(acc))


if __name__ == "__main__":
    sct = shape_classifier_trainer()
    sct.load_data()
    #sct.pca_plot()
    sct.pca_reduction(variance_thresh=0.95)
    sct.train_model(start=1000)
    sct.test_model(stop=1000)
