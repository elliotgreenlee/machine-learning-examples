"""
Elliot Greenlee

571 Project 3

April 3, 2017

knn and backpropagation neural network
"""

import numpy as np
import math


class KNN:
    def __init__(self, training_features, training_classes, testing_features, testing_classes, prior_probabilities, k, number_of_features, number_of_classes, minkowski_number):

        # Input data
        self.NUMBER_OF_FEATURES = number_of_features
        self.NUMBER_OF_CLASSES = number_of_classes

        self.training_features = training_features
        self.training_classes = training_classes

        self.testing_features = testing_features
        self.testing_classes = testing_classes

        self.prior_probabilities = prior_probabilities

        self.k = k

        self.minkowski_number = minkowski_number

        # Calculated data
        self.training_features_in_class = [0] * self.NUMBER_OF_CLASSES

        self.classes = None

        self.training_features_by_class()
        self.knn()

    def training_features_by_class(self):
        for sample in self.training_classes:
            self.training_features_in_class[sample-1] += 1

    '''knn'''
    def knn(self):
        classes = []
        for sample in self.testing_features:
            classes.append(self.decide_knn(sample))

        self.classes = classes

    @staticmethod
    def determine_distance_manhattan(sample, other):
        sample = np.transpose(np.mat(sample))
        other = np.transpose(np.mat(other))

        diff = sample - other

        distance = np.sum(diff)

        return distance

    @staticmethod
    def determine_distance_euclidean(sample, other):
        sample = np.transpose(np.mat(sample))
        other = np.transpose(np.mat(other))

        diff = sample - other
        distance = math.sqrt(np.dot(np.transpose(diff), diff))

        return distance

    # Other minkowski distances l_k(a,b) = (sum_{i=1}^{d}|a_i-b_i|^k)^{1/k}
    def determine_distance_minkowski(self, sample, other):
        sample = np.transpose(np.mat(sample))
        other = np.transpose(np.mat(other))

        diff = sample - other
        diff = np.absolute(diff)
        power = np.power(diff, self.minkowski_number)
        summed = np.sum(power)
        distance = np.power(summed, 1.0/self.minkowski_number)

        # TODO: convert distance to int

        return distance

    def decide_knn(self, sample):
        distances = []

        for other in self.training_features:
            distances.append(self.determine_distance_minkowski(sample, other))

        distances = np.asarray(distances)

        idx = distances.argsort()[::1]
        # leave for debugging: sorted_distances = distances[idx]
        sorted_training_classes = np.asarray(self.training_classes)[idx]

        # take the first self.k closest, add to counts for each class
        class_counts = [0] * self.NUMBER_OF_CLASSES
        for i in range(0, self.k):
            class_counts[sorted_training_classes[i]-1] += 1

        posteriors = [0] * self.NUMBER_OF_CLASSES
        for i in range(0, self.NUMBER_OF_CLASSES):
            if self.training_features_in_class[i] == 0:
                posteriors[i] = 0
            else:
                posteriors[i] = ((class_counts[i] * len(self.training_features)) / (1.0 * self.k * self.training_features_in_class[i])) * self.prior_probabilities[i]

        max_index = posteriors.index(max(posteriors))

        return 1 + max_index

