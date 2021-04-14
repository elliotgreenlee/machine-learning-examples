"""
Elliot Greenlee

571 Project 3

April 3, 2017

knn and backpropagation neural network
"""

import numpy as np

DEBUG_LEVEL = 1  # 0 = normal, 1 = debug


class Data:
    # Settings
    NUMBER_OF_FEATURES = 9
    NUMBER_OF_CLASSES = 7
    training_file_name = 'data/data.txt'
    cross_validation_indices_file_name = 'data/data_index.txt'

    def __init__(self):
        self.features = None
        self.types = None

        self.cross_validation_sets = None

        self.prior_probabilities = [0] * self.NUMBER_OF_CLASSES

    # Load in the data
    def load_data(self):
        with open(self.training_file_name) as f:
            X = []
            for line in f:
                words = line.split()
                for i in range(0, self.NUMBER_OF_FEATURES):
                    words[i] = float(words[i])
                X.append(words)

            X = np.asarray(X)
        self.features = X[:, 0:self.NUMBER_OF_FEATURES].astype(np.float)
        self.types = X[:, self.NUMBER_OF_FEATURES].astype(np.int)

    def load_cross_validation_sets(self):
        with open(self.cross_validation_indices_file_name) as f:
            sets = []
            for line in f:
                words = line.split()
                set = []
                for word in words:
                    if word != '0':
                        set.append(int(word))
                sets.append(set)

        self.cross_validation_sets = sets

    def prior_probability(self):
        class_counts = [0] * self.NUMBER_OF_CLASSES
        for type in self.types:
            class_counts[type-1] += 1

        for i in range(0, self.NUMBER_OF_CLASSES):
            self.prior_probabilities[i] = (1.0 * class_counts[i]) / len(self.types)

    def preprocess(self):

        # Read in the features and types
        self.load_data()

        if DEBUG_LEVEL:
            print 'Features'
            print self.features
            print ''
            print 'Types'
            print self.types

        # Read in cross val sets
        self.load_cross_validation_sets()

        if DEBUG_LEVEL:
            print 'Cross validation sets'
            print self.cross_validation_sets

        # Determine prior probabilities
        self.prior_probability()

        if DEBUG_LEVEL:
            print 'Prior Probabilities'
            print self.prior_probabilities







