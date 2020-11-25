import numpy as np
import cv2

class shape_classifier:

    def __init__(self, train=False, dpath='./src/ivr_assignment/mgc_model.npz'):
        self.save_path = dpath
        if train:
            self.priors = None
            self.means = None
            self.covs = None
            self.inv_covs = None
            self.eigenvectors = None
            self.data_mean = None
        else:
            npzfile = np.load(self.save_path)
            self.priors = npzfile['arr_0']
            self.means = npzfile['arr_1']
            self.covs = npzfile['arr_2']
            self.inv_covs = npzfile['arr_3']
            self.eigenvectors = npzfile['arr_4']
            self.data_mean = npzfile['arr_5']


    def save_state(self):
        np.savez(self.save_path,
                 self.priors,
                 self.means,
                 self.covs,
                 self.inv_covs,
                 self.eigenvectors,
                 self.data_mean)


    def fit(self, X, Y):
        classes = np.unique(Y)
        C = classes.shape[0]
        N, D = X.shape
        self.priors = np.zeros(C)
        self.means = np.zeros((C,D))
        self.covs = np.zeros((C,D,D))
        self.inv_covs = np.zeros((C,D,D))
        for i in range(C):
            c = classes[i]
            X_c = X[Y==c, :]
            self.priors[i] = X_c.shape[0] / N
            X_c_mean = np.mean(X_c, axis=0)
            X_c_cov = np.cov(X_c.transpose(), bias=True)
            self.means[i,:] = X_c_mean
            self.covs[i,:,:] = X_c_cov
            self.inv_covs[i,:,:] = np.linalg.inv(X_c_cov)


    def predict(self, X):
        C = self.priors.shape[0]
        N, D = X.shape
        logprobs = np.zeros((C,N))
        for c in range(C):
            logprobs[c,:] = self.log_p_x_c(X,c) + np.log(self.priors[c])
        predictions = np.zeros(N)
        predictions[logprobs[0,:] > logprobs[1,:]] = 0
        predictions[logprobs[1,:] > logprobs[0,:]] = 1
        return predictions


    def log_p_x_c(self, X, c):
        N, D = X.shape
        m = self.means[c,:]
        cov = self.covs[c,:,:]
        a = (X-m)
        b = self.inv_covs[c,:,:]
        c = (X-m).transpose()
        return -0.5*np.log(np.linalg.det(cov)) + (-0.5)*np.diag((np.matmul(a, np.matmul(b, c))))


    def p_c_x(self, X, c):
        numerator = np.exp(self.log_p_x_c(X, c))*self.priors[c]
        denominator = 0
        C = self.priors.shape[0]
        for c_it in range(C):
            denominator += np.exp(self.log_p_x_c(X, c_it))*self.priors[c_it]
        return numerator/denominator
    
    
    def normalize(self, X):
        tmp = (X/255.0)-self.data_mean
        if self.eigenvectors is not None and tmp.shape[1] != self.covs.shape[1]:
            # project data to pca plane if not projected
            tmp = np.matmul(tmp, self.eigenvectors.transpose())
        return tmp
