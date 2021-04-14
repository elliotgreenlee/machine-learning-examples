"""
Elliot Greenlee
COSC 528 Project 1
October 5, 2017
"""

import math
import numpy
from collections import Counter
from sklearn.feature_selection import f_regression, mutual_info_regression


class Data:
    def __init__(self):
        self.features = []
        self.r = []


class AutoData:
    def __init__(self, filename, number_of_features):
        self.NUMBER_OF_FEATURES = number_of_features
        self.data = None  # nested list of all data (instances x (NUMBER_OF_FEATURES + 1))

        self.features = None  # nested list of features (instances x NUMBER_OF_FEATURES)
        self.normalized_features = None  # nested list of z-normalized features (instances x NUMBER_OF_FEATURES)
        self.reduced_features = None  # nested list of pca-reduced features (instances x NUMBER_OF_FEATURES)
        self.r = None  # list of regression variables (1 x self.size)

        self.load(filename)

    def load(self, filename):
        data_file = open(filename, 'r')

        self.data = []
        for line in data_file:
            bad = False
            data = line.split()
            data_list = data[0: 1 + self.NUMBER_OF_FEATURES]  # regression variable and features
            for i, feature in enumerate(data_list):
                if feature is '?':
                    bad = True
                else:
                    data_list[i] = float(feature)
            if not bad:
                self.data.append(data_list)

        self.features = []
        self.r = []
        for instance in self.data:
            self.features.append(instance[1: 1 + self.NUMBER_OF_FEATURES])
            self.r.append(instance[0])  # set regression variable

    def size(self):
        return len(self.data)

    def means(self):
        means = [0] * (self.NUMBER_OF_FEATURES + 1)

        size = self.size()
        for instance in self.data:
            for i, feature in enumerate(instance):
                means[i] += feature

        for i, feature in enumerate(means):
            means[i] = feature / (size * 1.0)

        return means

    def minimums(self):
        minimums = ["?"] * (self.NUMBER_OF_FEATURES + 1)

        for instance in self.data:
            for i, feature in enumerate(instance):
                if minimums[i] is '?' or feature < minimums[i]:
                    minimums[i] = feature

        return minimums

    def maximums(self):
        maximums = ["?"] * (self.NUMBER_OF_FEATURES + 1)
        for instance in self.data:
            for i, feature in enumerate(instance):
                if maximums[i] is '?' or feature > maximums[i]:
                    maximums[i] = feature

        return maximums

    def standard_deviations(self):
        standard_deviations = [0] * (self.NUMBER_OF_FEATURES + 1)

        means = self.means()
        size = self.size()

        for instance in self.data:
            for i, feature in enumerate(instance):
                standard_deviations[i] += math.pow(feature - means[i], 2)

        for i, feature in enumerate(standard_deviations):
            standard_deviations[i] = math.sqrt(feature / (size * 1.0))

        return standard_deviations

    def variances(self):
        variances = [0] * (self.NUMBER_OF_FEATURES + 1)

        standard_deviations = self.standard_deviations()
        size = self.size()

        for i, feature in enumerate(standard_deviations):
            variances[i] = feature / (size * 1.0)

        return variances

    # Return modes of the discrete data (cylinders, model year, and origin)
    def modes(self):
        modes = ["?"] * (self.NUMBER_OF_FEATURES + 1)

        cylinders = []
        model_year = []
        origin = []
        for instance in self.data:
            cylinders.append(instance[1])
            model_year.append(instance[6])
            origin.append(instance[7])

        modes[1] = Counter(cylinders).most_common(1)[0]
        modes[6] = Counter(model_year).most_common(1)[0]
        modes[7] = Counter(origin).most_common(1)[0]

        return modes

    def f_test(self):
        self.normalize()
        f_test, p = f_regression(self.normalized_features, self.r)
        return f_test / numpy.max(f_test), p

    def mutual_info(self):
        self.normalize()
        mi = mutual_info_regression(self.normalized_features, self.r)
        return mi / numpy.max(mi)

    def stats(self):
        print("Size: {}".format(self.size()))
        print("")

        print("mpg, cylinders, displacement, horsepower, weight, acceleration, model year, origin")
        print("Means: {}".format(self.means()))
        print("Minimums: {}".format(self.minimums()))
        print("Maximums: {}".format(self.maximums()))
        print("Standard Deviations: {}".format(self.standard_deviations()))
        print("Variances: {}".format(self.variances()))
        print("Modes: {}".format(self.modes()))
        print("")

        print("cylinders, displacement, horsepower, weight, acceleration, model year, origin")
        f_values, p_values = self.f_test()
        print("p-values: {}".format(p_values))
        print("f-values: {}".format(f_values))
        print("Mutual Info: {}".format(self.mutual_info()))
        print("")

    def normalize(self):
        means = self.means()[1: 1 + self.NUMBER_OF_FEATURES]
        standard_deviations = self.standard_deviations()[1: 1 + self.NUMBER_OF_FEATURES]

        normalized_features = []
        for instance in self.features:
            row_features = []
            for feature, m, s_d in zip(instance, means, standard_deviations):
                row_features.append((feature - m) / (s_d * 1.0))
            normalized_features.append(row_features)

        self.normalized_features = normalized_features

    def pca_reduce(self, basis):
        if basis > self.NUMBER_OF_FEATURES:
            print("Basis {} cannot be larger than the number of features {}".format(basis, self.NUMBER_OF_FEATURES))
            exit(1)
            return

        self.normalize()

        # normalized means
        means = [0] * self.NUMBER_OF_FEATURES

        size = self.size()
        for instance in self.normalized_features:
            for i, feature in enumerate(instance):
                means[i] += feature

        for i, feature in enumerate(means):
            means[i] = feature / (size * 1.0)

        means = numpy.array([means])

        # covariances
        covariances = numpy.zeros((self.NUMBER_OF_FEATURES, self.NUMBER_OF_FEATURES))

        for instance in self.normalized_features:
            diff = numpy.array([instance]) - means
            covariances = numpy.add(covariances, numpy.dot(numpy.transpose(diff), diff))

        covariances = numpy.multiply(covariances, 1.0 / size)

        # find the eigenvalues and vectors of the matrix
        eigenvalues, eigenvectors = numpy.linalg.eig(covariances)

        # sort the eigenvalues and corresponding eigenvectors
        idx = eigenvalues.argsort()[::-1]  # sort and reverse
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.reduced_features = numpy.dot(numpy.array(self.normalized_features), numpy.transpose(eigenvectors[0:basis]))
