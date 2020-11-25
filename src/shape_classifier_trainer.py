#!/usr/bin/env python3

import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Local imports
from shape_classifier import shape_classifier
import constants

class shape_classifier_trainer:
    """Class containing methods to read image samples and train shape_classifier"""

    def __init__(self):
        # ---To be used when running from catkin_ws---
        self.data_dir = constants.ML_DATA_DIR
        self.sc = shape_classifier(train=True, dpath='./src/ivr_assignment/mgc_model.npz')
        # ---To be used when running from ivr_assignment---
        #self.data_dir = './images/ml_samples/'
        #self.sc = shape_classifier(train=True, dpath='./mgc_model.npz')

        # Initialize data holders
        self.X = None
        self.Y = None


    # Loads image data into data holders. Start and stop numbers can be specified
    # to select a particular range of image samples from each class.
    def load_data(self, start=0, stop=5527):
        classes = ['sphere', 'cuboid']
        C = 2
        N = C*(stop-start)
        # Samples are 30x30 pixels, hence dimensions are 30*30 = 900
        D = 30*30
        XY = np.zeros((N, D+1))
        for i in range(C):
            c = classes[i]
            # Load images for class c
            for j in range(start, stop):
                imgstr = self.data_dir + c + '_' + str(j) + '.png'
                img = cv2.imread(imgstr, cv2.IMREAD_GRAYSCALE)
                idx = i*(stop-start)+j
                XY[idx, :-1] = np.reshape(img, D)
                XY[idx, -1] = i
        # Shuffle the data and separate class labels from features
        np.random.shuffle(XY)
        self.X = XY[:, :-1]
        self.Y = XY[:, -1]
        # Normalize
        self.X /= 255.0
        Xmean = np.mean(self.X, axis=0)
        self.X -= Xmean
        self.sc.data_mean = Xmean


    # Visualise the data via plotting their projection on the 2d pca plane and
    # plotting cumulative variance to visualise the importance of principal components.
    def pca_plot(self):
        pca = PCA()
        pca.fit(self.X)
        labels = ['sphere', 'cuboid']
        igen = pca.components_
        # 2d pca plot of data
        for i in range(2):
            proj = np.matmul(self.X[self.Y == i, :], igen.transpose())
            plt.scatter(proj[:, 0], proj[:, 1], label=labels[i])
        plt.legend()
        plt.show()
        # Cumulative variance plot
        var = pca.explained_variance_ratio_
        cumvar = [var[0]]
        for i in range(1, len(var)):
            cumvar.append(cumvar[i-1]+var[i])
        plt.plot(range(1, len(var)+1), cumvar)
        plt.show()


    # Reduce dimensionality of data to minimum number of principal components
    # which account for a minimum total variance, specified by variance_thresh
    def pca_reduction(self, variance_thresh=0.95):
        pca = PCA()
        pca.fit(self.X)
        variances = pca.explained_variance_ratio_
        igen = pca.components_
        total_var = 0
        cutoff = -1
        for (i, v) in enumerate(variances):
            total_var += v
            if total_var >= variance_thresh:
                cutoff = i+1
                break
        self.X = np.matmul(self.X, igen[:cutoff, :].transpose())
        self.sc.eigenvectors = igen[:cutoff, :]
        print('Reduced to {} dimensions.'.format(cutoff))


    # Train the model on the data. Parameters start and stop can be
    # specified to select a training subset.
    def train_model(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = self.X.shape[0]
        self.sc.fit(self.X[start:stop, :], self.Y[start:stop])


    # Test the model and report the accuracy of the predictions. Parameters
    # start and stop can be specified to select a testing subset.
    def test_model(self, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = self.X.shape[0]
        predictions = self.sc.predict(self.X[start:stop, :])
        cm = confusion_matrix(y_true=self.Y[start:stop], y_pred=predictions)
        acc = np.sum(np.diag(cm)) / np.sum(cm) * 100
        print('Accuracy: {}%'.format(acc))

# When the code is executed, load the data and train a new model.
# Save the model's parameters so that it can be used without having to re-train.
if __name__ == "__main__":
    sct = shape_classifier_trainer()
    sct.load_data()
    #sct.pca_plot()
    sct.pca_reduction(variance_thresh=0.95)
    sct.train_model()
    sct.test_model()
    sct.sc.save_state()
