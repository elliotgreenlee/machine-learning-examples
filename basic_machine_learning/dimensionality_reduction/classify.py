import numpy as np
import math


class Classify:
    def __init__(self, Xtr_features, Xtr_classes, Xte_features, Xte_classes, prior_probability_no):
        self.NUMBER_OF_FEATURES = len(Xtr_features[0][0])

        # Original data
        self.Xtr_features = Xtr_features
        self.Xtr_classes = Xtr_classes

        self.Xte_features = Xte_features
        self.Xte_classes = Xte_classes

        # Class split data
        self.Xtr_features_no = None
        self.Xtr_features_yes = None
        self.Xte_features_no = None
        self.Xte_features_yes = None

        # General discriminant data
        self.means_no = None
        self.means_yes = None
        self.covariances_no = None
        self.covariances_yes = None
        self.prior_no = prior_probability_no
        self.prior_yes = 1 - prior_probability_no

        # Case 1
        self.case1_covariance = None
        self.case1_classes = None
        self.case1_accuracy = 0

        # Case 2
        self.case2_covariances = None
        self.case2_classes = None
        self.case2_accuracy = 0

        # Case 3
        self.case3_covariances_no = None
        self.case3_covariances_yes = None
        self.case3_classes = None
        self.case3_accuracy = 0

        # kNN
        self.k = 1
        self.knn_classes = None
        self.knn_accuracy = 0
        self.knn_accuracies = []

        # Do setup
        self.setup()
        self.determine_covariances_no()
        self.determine_covariances_yes()

        # Do case 1
        self.case_1()

        # Do case 2
        self.case_2()

        # Do case 3
        self.case_3()

        # Do kNN
        self.knn()

        # Determine accuracies for every method
        self.determine_accuracies()

        # Determine best knn k
        accuracies = []
        for k in range(1, 21):
            self.k = k
            self.knn()
            accuracies.append(self.determine_accuracy(self.knn_classes))

        self.knn_accuracies = accuracies

        self.best_k = max(self.knn_accuracies)

        # Vary prior probability
        self.case1_tprs = []
        self.case2_tprs = []
        self.case3_tprs = []
        self.knn_tprs = []

        self.case1_fprs = []
        self.case2_fprs = []
        self.case3_fprs = []
        self.knn_fprs = []
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
            self.knn()
            self.knn_tprs.append(self.determine_tpr(self.Xte_classes, self.knn_classes))
            self.knn_fprs.append(self.determine_fpr(self.Xte_classes, self.knn_classes))

    '''Separate features by class'''
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
        self.Xtr_features_no, self.Xtr_features_yes = self.separate_classes(self.Xtr_features, self.Xtr_classes)

    def separate_testing_classes(self):
        self.Xte_features_no, self.Xte_features_yes = self.separate_classes(self.Xte_features, self.Xte_classes)

    '''Determine determinant means'''
    def determine_means(self, features):
        means = np.zeros((1, self.NUMBER_OF_FEATURES))  # 1 x number of features vector of 0s

        # Iterate over all samples.
        for sample in features:
            means = np.add(means, sample)  # add across all features in a sample

        means = np.multiply(means, 1.0 / len(features))  # element-wise multiply
        return np.transpose(means)  # number of features x 1

    def determine_means_no(self):
        self.means_no = self.determine_means(self.Xtr_features_no)

    def determine_means_yes(self):
        self.means_yes = self.determine_means(self.Xtr_features_yes)

    def setup(self):
        self.separate_training_classes()
        self.separate_testing_classes()

        self.determine_means_no()
        self.determine_means_yes()

    '''Determine real covariance matrix'''
    def determine_covariances(self, features, means):
        covariance_matrix = np.zeros((self.NUMBER_OF_FEATURES, self.NUMBER_OF_FEATURES))
        means_transpose = np.transpose(means)

        for sample in features:
            diff = sample - means_transpose
            covariance_matrix = np.add(covariance_matrix, np.dot(np.transpose(diff), diff))

        covariance_matrix = np.multiply(covariance_matrix, 1.0 / len(features))
        return covariance_matrix

    def determine_covariances_no(self):
        self.covariances_no = self.determine_covariances(self.Xtr_features_no, self.means_no)
        return

    def determine_covariances_yes(self):
        self.covariances_yes = self.determine_covariances(self.Xtr_features_yes, self.means_yes)
        return

    def decide(self, sample, g_no, g_yes):
        no = g_no(sample)
        yes = g_yes(sample)

        if no > yes:
            return 0
        else:
            return 1

    '''Determine case 1 covariance approximation'''
    def case1_determine_covariance(self):
        # Calculate variance from the covariance matrices
        variance = np.sum(self.covariances_no) + np.sum(self.covariances_yes)
        variance /= self.NUMBER_OF_FEATURES * self.NUMBER_OF_FEATURES * 2
        self.case1_covariance = variance

    '''Discriminant case 1'''
    def case1_discriminant(self):
        classes = []
        for sample in self.Xte_features:
            classes.append(self.decide(sample, self.case1_g_no, self.case1_g_yes))

        self.case1_classes = classes

    def case1_g(self, sample, means, prior):
        sample = np.transpose(np.mat(sample))

        w = np.multiply(1.0 / self.case1_covariance, means)
        w0 = np.multiply(-1.0 / (2 * self.case1_covariance), np.dot(np.transpose(means), means)) + math.log(prior)
        g = np.dot(np.transpose(w), sample) + w0
        return g

    def case1_g_no(self, sample):
        return self.case1_g(sample, self.means_no, self.prior_no)

    def case1_g_yes(self, sample):
        return self.case1_g(sample, self.means_yes, self.prior_yes)

    def case_1(self):
        self.case1_determine_covariance()
        self.case1_discriminant()

    '''Determine case 2 covariances'''
    def case2_determine_covariances(self):
        # Calculate covariance matrix from covariance matrices
        covariances = np.add(self.covariances_no, self.covariances_yes)
        covariances = np.divide(covariances, 2)
        self.case2_covariances = covariances

    '''Discriminant case 2'''
    def case2_discriminant(self):
        classes = []
        for sample in self.Xte_features:
            classes.append(self.decide(sample, self.case2_g_no, self.case2_g_yes))

        self.case2_classes = classes

    def case2_g(self, sample, means, prior):
        sample = np.transpose(np.mat(sample))

        w = np.dot(np.linalg.inv(self.case2_covariances), means)
        w0 = np.multiply(-1.0/2.0, np.dot(np.dot(np.transpose(means), np.linalg.inv(self.case2_covariances)), means)) + math.log(prior)
        g = np.add(np.dot(np.transpose(w), sample), w0)
        return g

    def case2_g_no(self, sample):
        return self.case2_g(sample, self.means_no, self.prior_no)

    def case2_g_yes(self, sample):
        return self.case2_g(sample, self.means_yes, self.prior_yes)

    def case_2(self):
        self.case2_determine_covariances()
        self.case2_discriminant()

    '''Determine case 3 covariances'''
    def case3_determine_covariances(self):
        self.case3_covariances_no = self.covariances_no
        self.case3_covariances_yes = self.covariances_yes

    '''Discriminant case 3'''
    def case3_discriminant(self):
        classes = []
        for sample in self.Xte_features:
            classes.append(self.decide(sample, self.case3_g_no, self.case3_g_yes))

        self.case3_classes = classes

    def case3_g(self, sample, means, covariances, prior):
        sample = np.transpose(np.mat(sample))

        W = np.multiply(-1.0/2.0, np.linalg.inv(covariances))
        w = np.dot(np.linalg.inv(covariances), means)
        w0 = np.multiply(-1.0/2.0, np.dot(np.dot(np.transpose(means), np.linalg.inv(covariances)), means)) \
             - (1.0 / 2.0) * math.log(np.linalg.det(covariances)) + math.log(prior)
        g = np.dot(np.dot(np.transpose(sample), W), sample) + np.dot(np.transpose(w), sample) + w0
        return g

    def case3_g_no(self, sample):
        return self.case3_g(sample, self.means_no, self.case3_covariances_no, self.prior_no)

    def case3_g_yes(self, sample):
        return self.case3_g(sample, self.means_yes, self.case3_covariances_yes, self.prior_yes)

    def case_3(self):
        self.case3_determine_covariances()
        self.case3_discriminant()
        return

    '''knn'''
    def knn(self):
        classes = []
        for sample in self.Xte_features:
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

        for other in self.Xtr_features:
            distances.append(self.determine_distance(sample, other))

        distances = np.asarray(distances)

        idx = distances.argsort()[::1]
        # leave for debugging: sorted_distances = distances[idx]
        sorted_Xtr_classes = np.asarray(self.Xtr_classes)[idx]

        # take the first self.k closest, add to counts for each class
        no_count = 0
        yes_count = 0
        for i in range(0, self.k):
            if sorted_Xtr_classes[i] == 0:
                no_count += 1
            else:
                yes_count += 1

        posterior_no = ((no_count * len(self.Xtr_features)) / (self.k * len(self.Xtr_features_no))) * self.prior_no

        posterior_yes = ((yes_count * len(self.Xtr_features)) / (self.k * len(self.Xtr_features_yes))) * self.prior_yes

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






