import numpy as np


class FLD:
    def __init__(self, nXtr_features, Xtr_classes, nXte_features, Xte_classes):
        self.NUMBER_OF_FEATURES = 7
        self.w = None


        # Training data
        self.nXtr_features = nXtr_features
        self.Xtr_classes = Xtr_classes

        self.nXtr_features_no = None
        self.nXtr_means_no = None
        self.nXtr_covariances_no = None

        self.nXtr_features_yes = None
        self.nXtr_means_yes = None
        self.nXtr_covariances_yes = None

        self.fXtr_features = []  # fld reduced testing features

        # Testing data
        self.nXte_features = nXte_features
        self.Xte_classes = Xte_classes

        self.nXte_features_no = None
        self.nXte_features_yes = None

        self.fXte_features = []  # fld reduced testing features

        # Classification
        self.classify = None

        self.separate_training_classes()
        self.separate_testing_classes()

        self.determine_training_means()
        self.determine_training_covariances()

        self.determine_w()

        self.fld_training()
        self.fld_testing()

    '''Separate normalized features by class'''
    @staticmethod
    def separate_classes(features, classes):
        features_yes = []
        features_no = []
        for sample, type in zip(features, classes):
            if type == 0:
                features_no.append(sample)
            elif type == 1:
                features_yes.append(sample)
        return features_no, features_yes

    def separate_training_classes(self):
        self.nXtr_features_no, self.nXtr_features_yes = self.separate_classes(self.nXtr_features, self.Xtr_classes)

    def separate_testing_classes(self):
        self.nXte_features_no, self.nXte_features_yes = self.separate_classes(self.nXte_features, self.Xte_classes)

    '''Calculate class wise means'''
    def determine_means(self, features):
        means = np.zeros((1, self.NUMBER_OF_FEATURES))
        for sample in features:
            means = np.add(means, sample)

        means = np.multiply(means, 1.0 / len(features))
        return np.transpose(means)

    def determine_training_means(self):
        self.nXtr_means_no = self.determine_means(self.nXtr_features_no)
        self.nXtr_means_yes = self.determine_means(self.nXtr_features_yes)

    '''Calculate class wise covariance matrices'''
    def determine_covariance_matrix(self, features, means):
        covariance_matrix = np.zeros((self.NUMBER_OF_FEATURES, self.NUMBER_OF_FEATURES))
        means_transpose = np.transpose(means)
        for sample in features:
            diff = sample - means_transpose
            covariance_matrix = np.add(covariance_matrix, np.dot(np.transpose(diff), diff))

        covariance_matrix = np.multiply(covariance_matrix, 1.0 / len(features))
        return covariance_matrix

    def determine_training_covariances(self):
        self.nXtr_covariances_no = self.determine_covariance_matrix(self.nXtr_features_no, self.nXtr_means_no)
        self.nXtr_covariances_yes = self.determine_covariance_matrix(self.nXtr_features_yes, self.nXtr_means_yes)

    '''Determine w'''
    def determine_w(self):
        within_scatter = np.add(self.nXtr_covariances_no, self.nXtr_covariances_yes)
        diff = np.subtract(self.nXtr_means_no, self.nXtr_means_yes)
        self.w = np.dot(np.linalg.inv(within_scatter), diff)

    '''Reduce the feature dimensions using fld'''
    def fld(self, features):
        return np.dot(features, self.w)

    def fld_training(self):
        self.fXtr_features = self.fld(self.nXtr_features)

    def fld_testing(self):
        self.fXte_features = self.fld(self.nXte_features)
