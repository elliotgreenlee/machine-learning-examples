from ReadData import *
from MaximumLikelihoodEstimation import *
from Discriminant import *
from LikelihoodRatio import *
from DecisionRule import *
from TestingAccuracy import *
import numpy


def main():

    # Read in training data
    x_train, class_train = read_training_data()
    x_train = numpy.array(x_train)
    class_train = numpy.array(class_train)

    # Separate into classes
    class1_indices_train = numpy.where(class_train == 0)[0]
    x1_train = x_train[class1_indices_train]  # n x 2 matrix
    n1_train = len(class1_indices_train)

    class2_indices_train = numpy.where(class_train == 1)[0]
    x2_train = x_train[class2_indices_train]
    n2_train = len(class2_indices_train)

    # Determine mean and variance using maximum likelihood estimation
    sample_mean1 = find_sample_mean(n1_train, x1_train)
    sample_variance1 = find_sample_variance(n1_train, x1_train, sample_mean1)
    sample_mean2 = find_sample_mean(n2_train, x2_train)
    sample_variance2 = find_sample_variance(n2_train, x2_train, sample_mean2)

    print "sample mean of class 1", sample_mean1  # 1 x 2 matrix
    print "sample variance of class 1", sample_variance1  # 2 x 2 matrix
    print "sample mean of class 2", sample_mean2  # 1 x 2 matrix
    print "sample variance of class 2", sample_variance2  # 2 x 2 matrix

    # TODO: Plot means on train class data

# -------------------------------------------------------------------------------------------------------------------- #

    # Read in testing data.
    x_test, class_test = read_testing_data()
    x_test = numpy.array(x_test)
    class_test = numpy.array(class_test)

    # TODO: Plot data by itself

    class1_indices_test = numpy.where(class_test == 0)[0]
    x1_test = x_test[class1_indices_test]  # n x 2 matrix

    class2_indices_test = numpy.where(class_test == 1)[0]
    x2_test = x_test[class2_indices_test]  # n x 2 matrix

    # TODO: Plot class data

    # TODO: Plot means on test class data

# -------------------------------------------------------------------------------------------------------------------- #

    # Test accuracies at equal prior probability
    # Test likelihood function decision rule
    likelihoodAccuracy = likelihood_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2)
    print "Likelihood Testing Accuracy: ", likelihoodAccuracy

    # Test discriminant function decision rules
    # Case 1
    case1Accuracy = case_1_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2)
    print "Case 1 Testing Accuracy: ", case1Accuracy

    # TODO: this graph and print
    '''
    var = sample_variance1[0, 0] + sample_variance1[0, 1] + sample_variance1[1, 0] + sample_variance1[1, 1]
    var += + sample_variance2[0, 0] + sample_variance2[0, 1] + sample_variance2[1, 0] + sample_variance2[1, 1]
    var /= 8.0
    a, b = case_1_equation(sample_mean1, sample_mean2, var)
    print a, b
    x = numpy.linspace(-2, 2, 100)
    y = a * x + b
    print "The decision rule for case 1 is: y =", a, "x +", b

    #plt.plot(x, y, 'r', label="Discriminant Case 1")
    '''

    # Case 2
    case2Accuracy = case_2_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2)
    print "Case 2 Testing Accuracy: ", case2Accuracy

    # TODO: this graph and print
    '''
    var = numpy.add(sample_variance1, sample_variance2)
    var = numpy.divide(var, 2)
    a, b = case_2_equation(sample_mean1, sample_mean2, var)
    print a, b
    print "The decision rule for case 2 is: y =", a, "x +", b

    x = numpy.linspace(-2, 2, 100)
    y = a * x + b
    #plt.plot(x, y, 'k', label="Discriminant Case 2")
    '''

    # Case 3
    case3Accuracy = case_3_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2)
    print "Case 3 Testing Accuracy: ", case3Accuracy

    # TODO: this graph and print
    '''
    x1s, y1s, x2s, y2s = case_3_equation(sample_mean1, sample_mean2, sample_variance1, sample_variance2)

    a, b, c = numpy.polyfit(x2s, y2s, 2)
    print a, b, c
    print "The decision rule for case 3 can be approximated to: y =", a, "x^2 +", b, "x +", c

    #plt.plot(x1s, y1s, 'm')
    #plt.plot(x2s, y2s, 'm', label="Discriminant Case 3")

    #p = numpy.poly1d(numpy.polyfit(x2s, y2s, 2))
    #xp = numpy.linspace(-2, 2, 100)
    #plt.plot(xp, p(xp), 'm')
    '''

    # Test bimodal likelihood function decision rule
    bimodalAccuracy = bimodal_accuracy(x1_test, x2_test)
    print "Bimodal Likelihood Testing Accuracy: ", bimodalAccuracy

# -------------------------------------------------------------------------------------------------------------------- #

    # Determine accuracies for all probability distributions
    likelihood_accuracies = []
    case1_accuracies = []
    case2_accuracies = []
    case3_accuracies = []
    bimodal_accuracies = []
    prior_probabilities = []

    for prior_probability in range(1, 100):
        prior_probabilities.append(prior_probability)

        prior_probability1 = prior_probability / 100.0
        prior_probability2 = 1 - prior_probability1

        # Test likelihood function decision rule
        likelihoodAccuracy = likelihood_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2, prior_probability1, prior_probability2)
        likelihood_accuracies.append(likelihoodAccuracy)

        # Test discriminant function decision rules
        # Case 1
        case1Accuracy = case_1_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2, prior_probability1, prior_probability2)
        case1_accuracies.append(case1Accuracy)

        # Case 2
        case2Accuracy = case_2_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2, prior_probability1, prior_probability2)
        case2_accuracies.append(case2Accuracy)

        # Case 3
        case3Accuracy = case_3_accuracy(x1_test, x2_test, sample_mean1, sample_mean2, sample_variance1, sample_variance2, prior_probability1, prior_probability2)
        case3_accuracies.append(case3Accuracy)

        # Test bimodal likelihood function decision rule
        bimodalAccuracy = bimodal_accuracy(x1_test, x2_test, prior_probability1, prior_probability2)
        bimodal_accuracies.append(bimodalAccuracy)

    # Calculate the max accuracies
    print "The maximum accuracies when varying prior probability distribution were: "

    likelihood_index = likelihood_accuracies.index(max(likelihood_accuracies))
    print "Likelihood ratio:"
    print "(P0 =", prior_probabilities[likelihood_index], "%, P1 =", 100 - prior_probabilities[likelihood_index], "%):", likelihood_accuracies[likelihood_index]

    case1_index = case1_accuracies.index(max(case1_accuracies))
    print "Discriminant Case 1:"
    print "(P0 =", prior_probabilities[case1_index], "%, P1 =", 100 - prior_probabilities[case1_index], "%):", case1_accuracies[case1_index]

    case2_index = case2_accuracies.index(max(case2_accuracies))
    print "Discriminant Case 2:"
    print "(P0 =", prior_probabilities[case2_index], "%, P1 =", 100 - prior_probabilities[case2_index], "%):", case2_accuracies[case2_index]

    case3_index = case3_accuracies.index(max(case3_accuracies))
    print "Discriminant Case 3:"
    print "(P0 =", prior_probabilities[case3_index], "%, P1 =", 100 - prior_probabilities[case3_index], "%):", case3_accuracies[case3_index]

    bimodal_index = bimodal_accuracies.index(max(bimodal_accuracies))
    print "Bimodal:"
    print "(P0 =", prior_probabilities[bimodal_index], "%, P1 =", 100 - prior_probabilities[bimodal_index], "%):", bimodal_accuracies[bimodal_index]

    # Plot prior probabilities graphs
    plt.figure(1)
    plt.plot(prior_probabilities, likelihood_accuracies)
    plt.plot(prior_probabilities[likelihood_index], likelihood_accuracies[likelihood_index], 'rx', markersize=10, mew=4, label="Likelihood Ratio")
    plt.annotate('(%s, %s)' %(prior_probabilities[likelihood_index], likelihood_accuracies[likelihood_index]), xy=(prior_probabilities[likelihood_index] + 5, likelihood_accuracies[likelihood_index]))
    plt.title("Likelihood Ratio")
    plt.xlabel("Prior Probability 0")
    plt.ylabel("Accuracy")
    plt.axis([0, 100, 0, 1])
    plt.legend(loc=4, numpoints=1, prop={'size': 8})

    plt.figure(2)
    plt.plot(prior_probabilities, case1_accuracies)
    plt.plot(prior_probabilities[case1_index ], case1_accuracies[case1_index], 'rx', markersize=10, mew=4, label="Case 1")
    plt.annotate('(%s, %s)' % (prior_probabilities[case1_index], case1_accuracies[case1_index]),
                 xy=(prior_probabilities[case1_index] + 5, case1_accuracies[case1_index]))
    plt.title("Discriminant Case 1")
    plt.xlabel("Prior Probability 0")
    plt.ylabel("Accuracy")
    plt.axis([0, 100, 0, 1])
    plt.legend(loc=4, numpoints=1, prop={'size': 8})

    plt.figure(3)
    plt.plot(prior_probabilities, case2_accuracies)
    plt.plot(prior_probabilities[case2_index], case2_accuracies[case2_index], 'rx', markersize=10, mew=4, label="Case 2")
    plt.annotate('(%s, %s)' % (prior_probabilities[case2_index], case2_accuracies[case2_index]),
                 xy=(prior_probabilities[case2_index] + 5, case2_accuracies[case2_index]))
    plt.title("Discriminant Case 2")
    plt.xlabel("Prior Probability 0")
    plt.ylabel("Accuracy")
    plt.axis([0, 100, 0, 1])
    plt.legend(loc=4, numpoints=1, prop={'size': 8})

    plt.figure(4)
    plt.plot(prior_probabilities, case3_accuracies)
    plt.plot(prior_probabilities[case3_index], case3_accuracies[case3_index], 'rx', markersize=10, mew=4, label="Case 3")
    plt.annotate('(%s, %s)' % (prior_probabilities[case3_index], case3_accuracies[case3_index]),
                 xy=(prior_probabilities[case3_index] + 5, case3_accuracies[case3_index]))
    plt.title("Discriminant Case 3")
    plt.xlabel("Prior Probability 0")
    plt.ylabel("Accuracy")
    plt.axis([0, 100, 0, 1])
    plt.legend(loc=4, numpoints=1, prop={'size': 8})

    plt.figure(5)
    plt.plot(prior_probabilities, bimodal_accuracies)
    plt.plot(prior_probabilities[bimodal_index], bimodal_accuracies[bimodal_index], 'rx', markersize=10, mew=4, label="Bimodal")
    plt.annotate('(%s, %s)' % (prior_probabilities[bimodal_index], bimodal_accuracies[bimodal_index]),
                 xy=(prior_probabilities[bimodal_index] + 5, bimodal_accuracies[bimodal_index]))
    plt.title("Bimodal")
    plt.xlabel("Prior Probability 0")
    plt.ylabel("Accuracy")
    plt.axis([0, 100, 0, 1])
    plt.legend(loc=4, numpoints=1, prop={'size':8})

    plt.show()

    return

if __name__ == "__main__":
    main()