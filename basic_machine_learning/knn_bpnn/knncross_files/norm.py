"""
Elliot Greenlee

571 Project 3

April 3, 2017

knn and backpropagation neural network
"""

import numpy as np


class Norm:
    def __init__(self, training_features, testing_features, number_of_features):
        self.NUMBER_OF_FEATURES = number_of_features

        self.training_features = training_features
        self.testing_features = testing_features

        self.means = None  # NUMBER_OF_FEATURES x 1, vector of means
        self.sigmas = None  # NUMBER_OF_FEATURES x 1, vector of standard deviations

        # Normalized features
        self.normalized_training_features = None
        self.normalized_testing_features = None

        # Do normalization
        self.determine_means()
        self.determine_sigmas()

        self.normalize_training_data()
        self.normalize_testing_data()

    def determine_means(self):
        means = np.zeros((1, self.NUMBER_OF_FEATURES))
        for sample in self.training_features:
            means = np.add(means, sample)

        means = np.multiply(means, 1.0 / len(self.training_features))
        self.means = np.transpose(means)

    def determine_sigmas(self):
        means_transpose = np.transpose(self.means)
        sigmas = np.zeros((1, self.NUMBER_OF_FEATURES))

        for sample in self.training_features:
            diff = sample - means_transpose
            sigmas = np.add(sigmas, np.square(diff))

        sigmas = np.multiply(sigmas, 1.0 / len(self.training_features))
        sigmas = np.sqrt(sigmas)
        self.sigmas = np.transpose(sigmas)

    def normalize(self, features):
        normalized_features = []
        means_transpose = np.transpose(self.means)
        sigmas_transpose = np.transpose(self.sigmas)
        for sample in features:
            normalized_features.append(np.divide((sample - means_transpose), sigmas_transpose))

        return normalized_features

    def normalize_training_data(self):
        self.normalized_training_features = self.normalize(self.training_features)

    def normalize_testing_data(self):
        self.normalized_testing_features = self.normalize(self.testing_features)