import numpy as np
import math


class Classify:
    def __init__(self, training_features, testing_features, training_types, priors):
        self.NUMBER_OF_FEATURES = 44
        self.NUMBER_OF_TYPES = 4

        # Original data
        self.training_types = training_types
        self.testing_features = testing_features
        self.training_features = [[] for i in range(self.NUMBER_OF_TYPES)]
        for type, sample in zip(self.training_types, training_features):
            self.training_features[type].append(sample.tolist())

        self.means = [[] for i in range(self.NUMBER_OF_TYPES)]
        self.covariances = [[] for i in range(self.NUMBER_OF_TYPES)]
        self.priors = priors

        # Determine means and covariances
        self.setup()

        # Case 1
        self.case1_covariance = None
        self.case1_classes = None
        self.case1_accuracy = 0

        self.case_1()

        # Case 2
        self.case2_covariances = None
        self.case2_classes = None
        self.case2_accuracy = 0

        self.case_2()

        # Case 3
        self.case3_covariances = self.covariances
        self.case3_classes = None
        self.case3_accuracy = 0

        self.case_3()
        '''

        exit(1)

        # Determine accuracies for every method
        self.determine_accuracies()

        # Vary prior probability
        self.case1_tprs = []
        self.case2_tprs = []
        self.case3_tprs = []

        self.case1_fprs = []
        self.case2_fprs = []
        self.case3_fprs = []
        for prior_no in range(1, 99):
            prior_yes = 100 - prior_no
            prior_no /= 100.0
            prior_yes /= 100.0

            self.prior_no = prior_no
            self.prior_yes = prior_yes

            self.case_1()
            self.case1_tprs.append(self.determine_tpr(self.Xte_classes, self.case1_classes))
            self.case1_fprs.append(self.determine_fpr(self.Xte_classes, self.case1_classes))
            self.case_2()
            self.case2_tprs.append(self.determine_tpr(self.Xte_classes, self.case2_classes))
            self.case2_fprs.append(self.determine_fpr(self.Xte_classes, self.case2_classes))
            self.case_3()
            self.case3_tprs.append(self.determine_tpr(self.Xte_classes, self.case3_classes))
            self.case3_fprs.append(self.determine_fpr(self.Xte_classes, self.case3_classes))
        '''

    '''Determine determinant means'''
    def determine_means(self):

        for index, features in enumerate(self.training_features):
            means_sum = np.zeros((1, self.NUMBER_OF_FEATURES))  # 1 x number of features vector of 0s

            # Iterate over all samples.
            for sample in features:
                means_sum = np.add(means_sum, sample)  # add across all features in a sample

            self.means[index] = np.multiply(means_sum, 1.0 / len(features))[0]  # element-wise multiply

    '''Determine real covariance matrix'''
    def determine_covariances(self):
        for index, mf in enumerate(zip(self.means, self.training_features)):
            means = mf[0]
            features = mf[1]

            covariance_matrix = np.zeros((self.NUMBER_OF_FEATURES, self.NUMBER_OF_FEATURES))

            for sample in features:
                diff = np.subtract(sample, means)
                dot = np.dot(np.transpose(np.matrix(diff)), np.matrix(diff))
                covariance_matrix = np.add(covariance_matrix, dot)

            covariance_matrix = np.multiply(covariance_matrix, 1.0 / len(features))
            self.covariances[index] = covariance_matrix

    def setup(self):
        self.determine_means()
        self.determine_covariances()

    '''Determine case 1 covariance approximation'''
    def case1_determine_covariance(self):
        # Calculate variance from the covariance matrices
        variance = 0
        for covariance in self.covariances:
            variance += np.sum(covariance)

        variance /= self.NUMBER_OF_FEATURES * self.NUMBER_OF_FEATURES * self.NUMBER_OF_TYPES
        self.case1_covariance = variance

    '''Discriminant case 1'''
    def case1_discriminant(self):
        classes = []
        for sample in self.testing_features:
            posterior = [0] * self.NUMBER_OF_TYPES
            for type in range(self.NUMBER_OF_TYPES):
                posterior[type] = self.case1_g(sample, self.means[type], self.priors[type])

            classes.append(posterior.index(max(posterior)))

        self.case1_classes = classes

    def case1_g(self, sample, means, prior):
        sample = np.transpose(np.mat(sample))

        w = np.multiply(1.0 / self.case1_covariance, means)
        w0 = np.multiply(-1.0 / (2 * self.case1_covariance), np.dot(np.transpose(means), means)) + math.log(prior)
        g = np.dot(np.transpose(w), sample) + w0
        return g

    def case_1(self):
        self.case1_determine_covariance()
        self.case1_discriminant()

    '''Determine case 2 covariances'''
    def case2_determine_covariances(self):
        # Calculate covariance matrix from covariance matrices
        covariances = np.zeros([self.NUMBER_OF_FEATURES, self.NUMBER_OF_FEATURES])

        for type in range(self.NUMBER_OF_TYPES):
            covariances = np.add(covariances, self.covariances[type])

        covariances = np.divide(covariances, self.NUMBER_OF_TYPES)
        self.case2_covariances = covariances

    '''Discriminant case 2'''
    def case2_discriminant(self):
        classes = []
        for sample in self.testing_features:
            posterior = [0] * self.NUMBER_OF_TYPES
            for type in range(self.NUMBER_OF_TYPES):
                posterior[type] = self.case2_g(sample, self.means[type], self.priors[type])

            classes.append(posterior.index(max(posterior)))

        self.case2_classes = classes

    def case2_g(self, sample, means, prior):
        sample = np.transpose(np.mat(sample))
        means = np.transpose(np.mat(means))
        inv = np.linalg.pinv(self.case2_covariances)
        w = np.dot(inv, means)
        w0 = np.multiply(-1.0/2.0, np.dot(np.dot(np.transpose(means), inv), means)) + math.log(prior)
        dot = np.dot(np.transpose(w), sample)
        g = np.add(dot, w0)
        return g

    def case_2(self):
        self.case2_determine_covariances()
        self.case2_discriminant()

    '''Discriminant case 3'''
    def case3_discriminant(self):
        classes = []
        for sample in self.testing_features:
            posterior = [0] * self.NUMBER_OF_TYPES
            for type in range(self.NUMBER_OF_TYPES):
                posterior[type] = self.case3_g(sample, self.means[type], self.covariances[type], self.priors[type])

            classes.append(posterior.index(max(posterior)))

        self.case3_classes = classes

    def case3_g(self, sample, means, covariances, prior):
        sample = np.transpose(np.mat(sample))
        means = np.transpose(np.mat(means))
        inv = np.linalg.pinv(covariances)

        W = np.multiply(-1.0/2.0, inv)
        w = np.dot(inv, means)
        front = np.multiply(-1.0/2.0, np.dot(np.dot(np.transpose(means), inv), means))
        eig_values, eig_derp = np.linalg.eig(covariances)
        det = np.product(eig_values[eig_values > 1e-12])
        back = (1.0 / 2.0) * math.log(det) + math.log(prior)
        w0 = front - back
        g = np.dot(np.dot(np.transpose(sample), W), sample) + np.dot(np.transpose(w), sample) + w0
        return g

    def case_3(self):
        self.case3_discriminant()
        return

    '''knn'''
    def knn(self):
        classes = []
        for sample in self.testing_features:
            classes.append(self.decide_knn(sample))

        self.knn_classes = classes

    def determine_distance(self, sample, other):
        sample = np.transpose(np.mat(sample))
        other = np.transpose(np.mat(other))

        diff = sample - other
        distance = math.sqrt(np.dot(np.transpose(diff), diff))

        return distance

    def decide_knn(self, sample):
        distances = []

        for other in self.training_features:
            distances.append(self.determine_distance(sample, other))

        distances = np.asarray(distances)

        idx = distances.argsort()[::1]
        # leave for debugging: sorted_distances = distances[idx]
        sorted_Xtr_classes = np.asarray(self.classes)[idx]

        # take the first self.k closest, add to counts for each class
        no_count = 0
        yes_count = 0
        for i in range(0, self.k):
            if sorted_Xtr_classes[i] == 0:
                no_count += 1
            else:
                yes_count += 1

        posterior_no = ((no_count * len(self.training_features)) / (1.0 * self.k * len(self.Xtr_features_no))) * self.prior_no

        posterior_yes = ((yes_count * len(self.training_features)) / (1.0 * self.k * len(self.Xtr_features_yes))) * self.prior_yes

        if posterior_no > posterior_yes:
            return 0
        else:
            return 1

    def determine_accuracy(self, prediction_classes):
        goodcount = 0
        badcount = 0
        for real, predict in zip(self.Xte_classes, prediction_classes):
            if predict == real:
                goodcount += 1
            else:
                badcount += 1

        return (goodcount * 1.0) / (goodcount + badcount)

    def determine_accuracies(self):
        self.case1_accuracy = self.determine_accuracy(self.case1_classes)
        self.case2_accuracy = self.determine_accuracy(self.case2_classes)
        self.case3_accuracy = self.determine_accuracy(self.case3_classes)
        self.knn_accuracy = self.determine_accuracy(self.knn_classes)

    def determine_tpr(self, real_classes, predicted_classes):
        tp = 0
        p = 0
        for real_class, predicted_class in zip(real_classes, predicted_classes):
            if real_class == 1:
                p += 1
                if predicted_class == 1:
                    tp += 1

        tpr = (tp * 1.0) / p
        return tpr

    def determine_fpr(self, real_classes, predicted_classes):
        fp = 0
        n = 0
        for real_class, predicted_class in zip(real_classes, predicted_classes):
            if real_class == 0:
                n += 1
                if predicted_class == 1:
                    fp += 1

        fpr = (fp * 1.0) / n
        return fpr






