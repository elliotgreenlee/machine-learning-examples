from Discriminant import *
from LikelihoodRatio import *


# Test likelihood function decision rule
def likelihood_accuracy(x0, x1, mean0, mean1, covariance0, covariance1, prior_probability0=0.5, prior_probability1=0.5):
    i = 0
    for sample in x0:
        if gaussian_likelihood(sample, mean0, mean1, covariance0, covariance1, prior_probability0, prior_probability1):
            i += 1

    for sample in x1:
        if gaussian_likelihood(sample, mean1, mean0, covariance1, covariance0, prior_probability1, prior_probability0):
            i += 1

    accuracy = float(i) / (len(x0) + len(x1))
    return accuracy


# Test discriminant function decision rule case 1
def case_1_accuracy(x0, x1, mean0, mean1, covariance0, covariance1, prior_probability0=0.5, prior_probability1=0.5):
    i = 0
    for sample in x0:
        if discriminant_case1(sample, mean0, mean1, covariance0, covariance1, prior_probability0, prior_probability1):
            i += 1

    for sample in x1:
        if discriminant_case1(sample, mean1, mean0, covariance1, covariance0, prior_probability1, prior_probability0):
            i += 1

    accuracy = float(i) / (len(x0) + len(x1))
    return accuracy


# Test discriminant function decision rule case 2
def case_2_accuracy(x0, x1, mean0, mean1, covariance0, covariance1, prior_probability0=0.5, prior_probability1=0.5):
    i = 0
    for sample in x0:
        if discriminant_case2(sample, mean0, mean1, covariance0, covariance1, prior_probability0, prior_probability1):
            i += 1

    for sample in x1:
        if discriminant_case2(sample, mean1, mean0, covariance1, covariance0, prior_probability1, prior_probability0):
            i += 1

    accuracy = float(i) / (len(x0) + len(x1))
    return accuracy


# Test discriminant function decision rule case 3
def case_3_accuracy(x0, x1, mean0, mean1, covariance0, covariance1, prior_probability0=0.5, prior_probability1=0.5):
    i = 0
    for sample in x0:
        if discriminant_case3(sample, mean0, mean1, covariance0, covariance1, prior_probability0, prior_probability1):
            i += 1

    for sample in x1:
        if discriminant_case3(sample, mean1, mean0, covariance1, covariance0, prior_probability1, prior_probability0):
            i += 1

    accuracy = float(i) / (len(x0) + len(x1))
    return accuracy


# Test bimodal likelihood function decision rule
def bimodal_accuracy(x0, x1, prior_probability0=0.5, prior_probability1=0.5):
    # Estimated bimodal parameters for class 0
    mu1_1 = numpy.array([[-0.7], [0.25]])
    mu2_1 = numpy.array([[0.3], [0.3]])
    var1_1 = numpy.array([[0.08, 0], [0, 0.08]])
    var2_1 = numpy.array([[0.08, 0], [0, 0.08]])

    a1_1 = 0.5
    a2_1 = 1 - a1_1

    # Estimated bimodal parameters for class 1
    mu1_2 = numpy.array([[-0.4], [0.7]])
    mu2_2 = numpy.array([[0.5], [0.6]])
    var1_2 = numpy.array([[0.08, 0], [0, 0.08]])
    var2_2 = numpy.array([[0.08, 0], [0, 0.08]])
    a1_2 = 0.5
    a2_2 = 1 - a1_2

    max_accuracy = 0

    i = 0
    for sample in x0:
        if bimodal_gaussian_likelihood(sample, mu1_1, mu2_1, var1_1, var2_1, a1_1, a2_1, mu1_2, mu2_2, var1_2, var2_2, a1_2,
                                       a2_2, prior_probability0, prior_probability1):
            i += 1

    for sample in x1:
        if bimodal_gaussian_likelihood(sample, mu1_2, mu2_2, var1_2, var2_2, a1_2, a2_2, mu1_1, mu2_1, var1_1, var2_1, a1_1,
                                       a2_1, prior_probability1, prior_probability0):
            i += 1

    accuracy = float(i) / (len(x0) + len(x1))

    return accuracy

