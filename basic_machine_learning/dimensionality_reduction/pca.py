import numpy as np


class PCA:
    def __init__(self, nXtr_features, nXte_features):
        self.NUMBER_OF_FEATURES = 7
        self.MAX_ERROR = 0.1

        self.nXtr_features = nXtr_features
        self.nXte_features = nXte_features

        self.covariance = None
        self.means = None

        self.eigenvalues = None
        self.eigenvectors = None

        self.E = None

        self.pXtr_features = None  # pca reduced training features
        self.pXte_features = None  # pca reduced testing features

        # Classification
        self.classify = None

        self.determine_means()
        self.determine_covariance()

        self.determine_eigenvectors()

        self.determine_e()

        self.pca_training()
        self.pca_testing()

    def determine_means(self):
        means = np.zeros((1, self.NUMBER_OF_FEATURES))
        for sample in self.nXtr_features:
            means = np.add(means, sample)

        means = np.multiply(means, 1.0 / len(self.nXtr_features))
        self.means = np.transpose(means)

    # find covariance matrix of all features
    def determine_covariance(self):
        covariance_matrix = np.zeros((self.NUMBER_OF_FEATURES, self.NUMBER_OF_FEATURES))
        means_transpose = np.transpose(self.means)
        for sample in self.nXtr_features:
            diff = sample - means_transpose
            covariance_matrix = np.add(covariance_matrix, np.dot(np.transpose(diff), diff))

        covariance_matrix = np.multiply(covariance_matrix, 1.0 / len(self.nXtr_features))
        self.covariance = covariance_matrix

    def determine_eigenvectors(self):
        # find the eigenvalues and vectors of the matrix
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance)

        # sort the eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.eigenvectors = eigenvectors[:, idx]

    def error_rate(self, remove_x):
        top = np.sum(self.eigenvalues[self.NUMBER_OF_FEATURES - remove_x : self.NUMBER_OF_FEATURES])
        bottom = np.sum(self.eigenvalues)

        return top / bottom

    def determine_e(self):
        # for each discarding possibility
        for i in range(0, self.NUMBER_OF_FEATURES):
            if self.error_rate(i) <= self.MAX_ERROR:
                # get first self.NUMBER_OF_FEATURES - i eigenvectors
                self.E = self.eigenvectors[0 : self.NUMBER_OF_FEATURES - i]

    # Reduce the feature dimensions using pca
    def pca(self, features):
        return np.dot(features, np.transpose(self.E))

    def pca_training(self):
        self.pXtr_features = self.pca(self.nXtr_features)

    def pca_testing(self):
        self.pXte_features = self.pca(self.nXte_features)
