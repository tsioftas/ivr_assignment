import numpy as np

class shape_classifier:
    """Class containing methods for predicting if a shape is a sphere or cuboid.
    The classifier used is a Multivariate Gaussian Classifier"""

    # Parameter train=True may be given to train a new model, otherwise the
    # parameters are loaded from the file specified by parameter dpath. The default
    # value for dpath works if the cwd is catkin_ws.
    def __init__(self, train=False, dpath='./src/ivr_assignment/mgc_model.npz'):
        self.save_path = dpath
        # Initialize / Load model parameters
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


    # Saves the model's parameters in the savefile
    def save_state(self):
        np.savez(self.save_path,
                 self.priors,
                 self.means,
                 self.covs,
                 self.inv_covs,
                 self.eigenvectors,
                 self.data_mean)


    # Train a new model
    def fit(self, X, Y):
        classes = np.unique(Y)
        C = classes.shape[0]
        N, D = X.shape
        self.priors = np.zeros(C)
        self.means = np.zeros((C, D))
        self.covs = np.zeros((C, D, D))
        self.inv_covs = np.zeros((C, D, D))
        for i in range(C):
            c = classes[i]
            X_c = X[Y == c, :]
            self.priors[i] = X_c.shape[0] / N
            X_c_mean = np.mean(X_c, axis=0)
            X_c_cov = np.cov(X_c.transpose(), bias=True)
            self.means[i, :] = X_c_mean
            self.covs[i, :, :] = X_c_cov
            self.inv_covs[i, :, :] = np.linalg.inv(X_c_cov)


    # Predict class labels for data in X. X has to be normalized and two dimensional.
    def predict(self, X):
        C = self.priors.shape[0]
        N, _ = X.shape
        # Doesn't actually compute probabilites, just the logarithm of numerator:
        # log(p(x|c)*p(c)) = log(p(x|c)) + log(p(c))
        logprobs = np.zeros((C, N))
        for c in range(C):
            logprobs[c, :] = self.log_p_x_c(X, c) + np.log(self.priors[c])
        predictions = np.zeros(N)
        predictions[logprobs[0, :] > logprobs[1, :]] = 0
        predictions[logprobs[1, :] > logprobs[0, :]] = 1
        return predictions


    # Calculate log(p(x|c)) for all data in X
    def log_p_x_c(self, X, c):
        m = self.means[c, :]
        cov = self.covs[c, :, :]
        a = (X-m)
        b = self.inv_covs[c, :, :]
        c = (X-m).transpose()
        return -0.5*np.log(np.linalg.det(cov)) + (-0.5)*np.diag((np.matmul(a, np.matmul(b, c))))


    # Calcualate p(c|x) for all data in X. This is more costly than the approach used in predict
    # thus it should only used as a tiebreaker.
    def p_c_x(self, X, c):
        numerator = np.exp(self.log_p_x_c(X, c))*self.priors[c]
        denominator = 0
        C = self.priors.shape[0]
        for c_it in range(C):
            denominator += np.exp(self.log_p_x_c(X, c_it))*self.priors[c_it]
        return numerator/denominator


    # Normalize data in X according to the training data and
    # project to pca plane used by the classifier
    def normalize(self, X):
        tmp = (X/255.0)-self.data_mean
        if self.eigenvectors is not None and tmp.shape[1] != self.covs.shape[1]:
            # project data to pca plane if not projected
            tmp = np.matmul(tmp, self.eigenvectors.transpose())
        return tmp
