import numpy as np


class Norm:
    def __init__(self, features):
        self.NUMBER_OF_FEATURES = 44

        self.original_features = features

        self.means = None  # NUMBER_OF_FEATURES x 1, vector of means
        self.sigmas = None  # NUMBER_OF_FEATURES x 1, vector of standard deviations

        self.features = None

        # Do normalization
        self.determine_means()
        self.determine_sigmas()

        self.normalize_data()

    def determine_means(self):
        means = np.zeros((1, self.NUMBER_OF_FEATURES))
        for sample in self.original_features:
            means = np.add(means, sample)

        self.means = np.multiply(means, 1.0 / len(self.original_features))

    def determine_sigmas(self):
        means_transpose = np.transpose(self.means)
        sigmas = np.zeros((1, self.NUMBER_OF_FEATURES))

        for sample in self.original_features:
            diff = sample - means_transpose
            sigmas = np.add(sigmas, np.square(diff))

        sigmas = np.multiply(sigmas, 1.0 / len(self.original_features))
        self.sigmas = np.sqrt(sigmas)

    def normalize(self, features):

        for i, sample in enumerate(features):
                diff = sample - self.means[0]
                div = np.divide(diff, self.sigmas[0])
                features[i] = div

        return features

    def normalize_data(self):
        self.features = self.normalize(self.original_features)
