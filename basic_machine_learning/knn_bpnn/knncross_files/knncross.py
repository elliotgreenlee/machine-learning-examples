"""
Elliot Greenlee

571 Project 3

April 3, 2017

knn and backpropagation neural network
"""

from data import *
from norm import *
from knn import *
import matplotlib.pyplot as plt

NUMBER_OF_SETS = 10


def main():
    data = Data()
    data.preprocess()

    # Set up for cross validation
    cv_data = []
    setnum = 1
    for set_indices in data.cross_validation_sets:
        if DEBUG_LEVEL:
            print 'cross validation set:', setnum
        setnum += 1
        # Separate into training and testing sets
        i = 0
        cv_training_features = []
        cv_training_classes = []
        cv_testing_features = []
        cv_testing_classes = []
        for feature, type in zip(data.features, data.types):
            i += 1
            if i in set_indices:
                cv_testing_features.append(feature)
                cv_testing_classes.append(type)
            else:
                cv_training_features.append(feature)
                cv_training_classes.append(type)

        if DEBUG_LEVEL:
            print 'cv_training_features'
            print cv_training_features
            print 'cv_training_classes'
            print cv_training_classes
            print 'cv_testing_features'
            print cv_testing_features
            print 'cv_testing_classes '
            print cv_testing_classes

        # Normalize for this set
        norm = Norm(cv_training_features, cv_testing_features, data.NUMBER_OF_FEATURES)
        cv_set = (norm.normalized_training_features, cv_training_classes, norm.normalized_testing_features, cv_testing_classes)
        cv_data.append(cv_set)

    # Determine best k
    print 'Find best knn k value'
    total_accuracies = [0] * 20
    i = 0
    for set in cv_data:
        i += 1
        if DEBUG_LEVEL:
            print 'Running set:', i

        accuracies = best_k_accuracies(set[0], set[1], set[2], set[3], data.prior_probabilities, data.NUMBER_OF_FEATURES, data.NUMBER_OF_CLASSES)
        if DEBUG_LEVEL:
            print accuracies

        for j in range(0, 20):
            total_accuracies[j] += accuracies[j]

    average_accuracies = [0] * 20

    for i in range(0, 20):
        average_accuracies[i] = total_accuracies[i] / NUMBER_OF_SETS * 1.0

    if DEBUG_LEVEL:
        print 'Average accuracies'
        print average_accuracies

    best_k = 1 + average_accuracies.index(max(average_accuracies))

    print 'Best k: ', best_k

    # TODO: graph all accuracies by k with best_k highlighted

    plt.figure(1)
    plt.plot(list(range(1, 21)), average_accuracies, 'rx', markersize=5)
    plt.title("kNN Average Accuracies vs. k Value")
    plt.xlabel("k Values")
    plt.ylabel("Accuracy")
    plt.axis([0, 21, 0, 1])

    print 'Find best Minkowski Distance'
    minkowski_max = 16
    total_accuracies = [0] * minkowski_max
    i = 0
    for set in cv_data:
        i += 1
        if DEBUG_LEVEL:
            print 'Running set:', i
        # Vary minkowski distance
        accuracies = []
        for j in range(1, minkowski_max+1):
            minkowski_number = j
            knn = KNN(set[0], set[1], set[2], set[3], data.prior_probabilities, best_k,
                      data.NUMBER_OF_FEATURES, data.NUMBER_OF_CLASSES, minkowski_number)
            accuracies.append(determine_accuracy(set[3], knn.classes))

        if DEBUG_LEVEL:
            print accuracies

        for j in range(0, minkowski_max):
            total_accuracies[j] += accuracies[j]

    average_accuracies = [0] * minkowski_max

    for i in range(0, minkowski_max):
        average_accuracies[i] = total_accuracies[i] / NUMBER_OF_SETS * 1.0

    if DEBUG_LEVEL:
        print 'Average accuracies'
        print average_accuracies

    best_minkowski = average_accuracies.index(max(average_accuracies))

    print 'Best minkowski: ', best_minkowski

    # TODO graph all accuracies by minkowski with best_minkowski highlighted

    plt.figure(2)
    plt.plot(list(range(1, minkowski_max+1)), average_accuracies, 'rx', markersize=5)
    plt.title("kNN Average Accuracies vs. Minkowski Distance Metric")
    plt.xlabel("Minkowski Distance")
    plt.ylabel("Accuracy")
    plt.axis([0, 17, 0, 1])

    plt.show()


def best_k_accuracies(training_features, training_classes, testing_features, testing_classes, prior_probabilities,
                      number_of_features, number_of_classes):
    # Determine best knn k
    accuracies = []
    for k in range(1, 21):
        knn = KNN(training_features, training_classes, testing_features, testing_classes, prior_probabilities, k,
                  number_of_features, number_of_classes, 2)
        accuracies.append(determine_accuracy(testing_classes, knn.classes))

    return accuracies


# performance calculations
def determine_accuracy(testing_classes, prediction_classes):
    goodcount = 0
    badcount = 0
    for real, predict in zip(testing_classes, prediction_classes):
        if predict == real:
            goodcount += 1
        else:
            badcount += 1

    return (goodcount * 1.0) / (goodcount + badcount)

if __name__ == "__main__":
    main()
