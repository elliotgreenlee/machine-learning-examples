"""
Elliot Greenlee
571 Project 2
March 10, 2017

2 class classification
"""

import numpy as np

class Data:
    # Settings
    NUMBER_OF_FEATURES = 44

    def __init__(self, input_files):
        self.input_files = input_files

        # Original training data
        self.features = None
        self.classes = None

        self.feature_set = []

    '''Load in the data'''
    def load_data(self, file_name):
        with open(file_name) as f:
            header = f.readline()  # skip the header row
            header = header.split()
            self.feature_set.append(int(header[6]))
            X = []
            for line in f:
                words = line.split()
                for i in range(0, self.NUMBER_OF_FEATURES):
                    words[i] = float(words[i])
                X.append(words)

            X = np.asarray(X)
        return X[:, 0:self.NUMBER_OF_FEATURES].astype(np.float), X[:, self.NUMBER_OF_FEATURES].astype(np.int)

    def load_10_data(self):
        new_features, new_classes = self.load_data(self.input_files.format(0))
        self.classes = new_classes
        self.features = new_features
        for i in range(1, 10):

            new_features, new_classes = self.load_data(self.input_files.format(i))
            self.classes = np.concatenate((self.classes, new_classes), axis=0)
            self.features = np.concatenate((self.features, new_features), axis=0)

    def load_separate_data(self):
        self.classes = [0] * 10
        self.features = [0] * 10
        for i in range(0, 10):
            new_features, new_classes = self.load_data(self.input_files.format(i))
            self.classes[i] = new_classes
            self.features[i] = new_features
