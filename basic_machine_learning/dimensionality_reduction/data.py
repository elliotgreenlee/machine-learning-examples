"""
Elliot Greenlee
571 Project 2
March 10, 2017

2 class classification
"""

import numpy as np
from norm import Norm
from pca import PCA
from fld import FLD
from classify import Classify
import matplotlib.pyplot as plt


class Data:
    # Settings
    NUMBER_OF_FEATURES = 7
    training_file_name = 'data/pima.tr'
    testing_file_name = 'data/pima.te'

    def __init__(self):
        # Original training data
        self.Xtr_feature_names = None
        self.Xtr_features = None
        self.Xtr_classes = None
        self.prior_probability_no = 0.5
        self.prior_probability_yes = 0.5

        # Original testing features
        self.Xte_feature_names = None
        self.Xte_features = None
        self.Xte_classes = None

        self.norm = None  # normalized
        self.pca = None  # pca reduced
        self.fld = None  # fld reduced

    '''Check errors with input data'''
    # Find errors in the loaded data'''
    def find_errors(self):
        for sample in self.Xtr_features:
            if len(sample) != self.NUMBER_OF_FEATURES:  # features
                print 'Error with training feature data: ' + sample

        for sample in self.Xte_features:
            if len(sample) != self.NUMBER_OF_FEATURES:  # features
                print 'Error with testing feature data: ' + sample

        if self.Xtr_feature_names != self.Xte_feature_names:
            print '<', self.Xtr_feature_names, '>', '<', self.Xte_feature_names, '>'
            print 'Training and testing features names do not agree'

    '''Load in the data'''
    def load_data(self, file_name):
        with open(file_name) as f:
            feature_names = f.readline()  # skip the header row
            X = []
            for line in f:
                words = line.split()
                for i in range(0, self.NUMBER_OF_FEATURES):
                    words[i] = float(words[i])
                X.append(words)

            X = np.asarray(X)
        return feature_names, X[:, 0:7].astype(np.float), X[:, 7]

    def load_training_data(self):
        self.Xtr_feature_names, self.Xtr_features, self.Xtr_classes = self.load_data(self.training_file_name)

    def load_testing_data(self):
        self.Xte_feature_names, self.Xte_features, self.Xte_classes = self.load_data(self.testing_file_name)

    '''Print the original data'''
    @staticmethod
    def print_data(x_features, x_class):
        for sample_features, sample_class in zip(x_features, x_class):
            print sample_features, sample_class

    def print_training_data(self):
        print 'Training Data'
        print self.Xtr_feature_names
        self.print_data(self.Xtr_features, self.Xtr_classes)
        print ''

    def print_testing_data(self):
        print 'Testing Data'
        print self.Xte_feature_names
        self.print_data(self.Xte_features, self.Xte_classes)
        print ''

    '''Convert class string data to booleans'''
    @staticmethod
    def boolify_class(x_class):
        boolified_x_class = []
        for sample in x_class:
            if sample == 'Yes':
                boolified_x_class.append(1)
            elif sample == 'No':
                boolified_x_class.append(0)
            else:
                print 'Error with class data: ' + sample
        return boolified_x_class

    def boolify_training_class(self):
        self.Xtr_classes = self.boolify_class(self.Xtr_classes)

    def boolify_testing_class(self):
        self.Xte_classes = self.boolify_class(self.Xte_classes)

    def determine_prior_probability(self):
        no_count = 0
        yes_count = 0
        for sample in self.Xtr_classes:
            if sample == 0:
                no_count += 1
            elif sample == 1:
                yes_count += 1

        self.prior_probability_no = (1.0 * no_count) / len(self.Xtr_classes)
        self.prior_probability_yes = (1.0 * yes_count) / len(self.Xtr_classes)

    '''Preprocess the data'''
    def preprocess(self):
        # Load the data
        data.load_training_data()
        data.load_testing_data()

        # Check for feature errors
        data.find_errors()

        # Change 'Yes' and 'No' to 1 and 0 indicating 'with disease' and 'without disease'
        self.boolify_training_class()
        self.boolify_testing_class()

        self.determine_prior_probability()

# Initialize
data = Data()

# Prepocess data
data.preprocess()

# Normalize data
data.norm = Norm(data.Xtr_features, data.Xte_features)

# Reduce data
data.pca = PCA(data.norm.nXtr_features, data.norm.nXte_features)
data.fld = FLD(data.norm.nXtr_features, data.Xtr_classes, data.norm.nXte_features, data.Xte_classes)

# Perform classification
data.norm.classify = Classify(data.norm.nXtr_features, data.Xtr_classes, data.norm.nXte_features, data.Xte_classes, data.prior_probability_no)
data.pca.classify = Classify(data.pca.pXtr_features, data.Xtr_classes, data.pca.pXte_features, data.Xte_classes, data.prior_probability_no)
data.fld.classify = Classify(data.fld.fXtr_features, data.Xtr_classes, data.fld.fXte_features, data.Xte_classes, data.prior_probability_no)

print 'Accuracies'
print 'Normalized'
print data.norm.classify.case1_accuracy
print data.norm.classify.case2_accuracy
print data.norm.classify.case3_accuracy
print data.norm.classify.knn_accuracy

print 'PCA'
print data.pca.classify.case1_accuracy
print data.pca.classify.case2_accuracy
print data.pca.classify.case3_accuracy
print data.pca.classify.knn_accuracy

print 'FLD'
print data.fld.classify.case1_accuracy
print data.fld.classify.case2_accuracy
print data.fld.classify.case3_accuracy
print data.fld.classify.knn_accuracy

print 'kNN varied'
print 'norm'
print data.norm.classify.knn_accuracies
plt.figure(1)
plt.plot(list(range(1, 21)), data.norm.classify.knn_accuracies, 'rx', markersize=4, label="Normalized")
plt.title("Normalized kNN Accuracies")
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.axis([0, 21, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})
print 'pca'
print data.pca.classify.knn_accuracies
plt.figure(2)
plt.plot(list(range(1, 21)), data.pca.classify.knn_accuracies, 'rx', markersize=4, label="PCA")
plt.title("PCA kNN Accuracies")
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.axis([0, 21, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})
print 'fld'
print data.fld.classify.knn_accuracies
plt.figure(3)
plt.plot(list(range(1, 21)), data.fld.classify.knn_accuracies, 'rx', markersize=4, label="FLD")
plt.title("FLD kNN Accuracies")
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.axis([0, 21, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print 'prior probability varied'
print 'norm'
print data.norm.classify.case1_tprs
print data.norm.classify.case1_fprs
plt.figure(4)
plt.plot(data.norm.classify.case1_fprs, data.norm.classify.case1_tprs, 'rx', markersize=4, label="Normalized Case 1")
plt.title("Normalized Case 1 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.norm.classify.case2_tprs
print data.norm.classify.case2_fprs
plt.figure(5)
plt.plot(data.norm.classify.case2_fprs, data.norm.classify.case2_tprs, 'rx', markersize=4, label="Normalized Case 2")
plt.title("Normalized Case 2 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.norm.classify.case3_tprs
print data.norm.classify.case3_fprs
plt.figure(6)
plt.plot(data.norm.classify.case3_fprs, data.norm.classify.case3_tprs, 'rx', markersize=4, label="Normalized Case 3")
plt.title("Normalized Case 3 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.norm.classify.knn_tprs
print data.norm.classify.knn_fprs
plt.figure(7)
plt.plot(data.norm.classify.knn_fprs, data.norm.classify.knn_tprs, 'rx', markersize=4, label="Normalized kNN")
plt.title("Normalized kNN ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print 'pca'
print data.pca.classify.case1_tprs
print data.pca.classify.case1_fprs
plt.figure(8)
plt.plot(data.pca.classify.case1_fprs, data.pca.classify.case1_tprs, 'rx', markersize=4, label="PCA Case 1")
plt.title("PCA Case 1 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.pca.classify.case2_tprs
print data.pca.classify.case2_fprs
plt.figure(9)
plt.plot(data.pca.classify.case2_fprs, data.pca.classify.case2_tprs, 'rx', markersize=4, label="PCA Case 2")
plt.title("PCA Case 2 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.pca.classify.case3_tprs
print data.pca.classify.case3_fprs
plt.figure(10)
plt.plot(data.pca.classify.case3_fprs, data.pca.classify.case3_tprs, 'rx', markersize=4, label="PCA Case 3")
plt.title("PCA Case 3 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.pca.classify.knn_tprs
print data.pca.classify.knn_fprs
plt.figure(11)
plt.plot(data.pca.classify.knn_fprs, data.pca.classify.knn_tprs, 'rx', markersize=4, label="PCA kNN")
plt.title("PCA kNN ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print 'fld'
print data.fld.classify.case1_tprs
print data.fld.classify.case1_fprs
plt.figure(12)
plt.plot(data.fld.classify.case1_fprs, data.fld.classify.case1_tprs, 'rx', markersize=4, label="FLD Case 1")
plt.title("FLD Case 1 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.fld.classify.case2_tprs
print data.fld.classify.case2_fprs
plt.figure(13)
plt.plot(data.fld.classify.case2_fprs, data.fld.classify.case2_tprs, 'rx', markersize=4, label="FLD Case 2")
plt.title("FLD Case 2 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.fld.classify.case3_tprs
print data.fld.classify.case3_fprs
plt.figure(14)
plt.plot(data.fld.classify.case3_fprs, data.fld.classify.case3_tprs, 'rx', markersize=4, label="FLD Case 3")
plt.title("FLD Case 3 ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print data.fld.classify.knn_tprs
print data.fld.classify.knn_fprs
plt.figure(15)
plt.plot(data.fld.classify.kNN_fprs, data.fld.classify.kNN_tprs, 'rx', markersize=4, label="FLD kNN")
plt.title("FLD kNN ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

plt.show()


