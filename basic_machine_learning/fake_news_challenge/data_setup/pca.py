import numpy as np


class PCA:
    def __init__(self, input_features, max_error):
        self.NUMBER_OF_FEATURES = 44
        self.MAX_ERROR = max_error

        self.input_features = input_features

        self.covariance = None
        self.means = None

        self.eigenvalues = None
        self.eigenvectors = None

        self.E = None

        self.features = None  # pca reduced training features

        self.determine_means()
        self.determine_covariance()

        self.determine_eigenvectors()

        self.determine_e()

        self.pca()

    def determine_means(self):
        means = np.zeros((1, self.NUMBER_OF_FEATURES))
        for sample in self.input_features:
            means = np.add(means, sample)

        means = np.multiply(means, 1.0 / len(self.input_features))
        self.means = np.transpose(means)

    # find covariance matrix of all features
    def determine_covariance(self):
        covariance_matrix = np.zeros((self.NUMBER_OF_FEATURES, self.NUMBER_OF_FEATURES))
        means_transpose = np.transpose(self.means)
        for sample in self.input_features:
            diff = sample - means_transpose
            covariance_matrix = np.add(covariance_matrix, np.dot(np.transpose(diff), diff))

        covariance_matrix = np.multiply(covariance_matrix, 1.0 / len(self.input_features))
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
    def pca(self):
        self.features = np.dot(self.input_features, np.transpose(self.E))
