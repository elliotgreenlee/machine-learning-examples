# Elliot Greenlee
# 571 Project 1
# February 7, 2016

# TODO: For this file, I need to put in the formulas for maximum likelihood estimation in the book.
# maybe I should also ask if I need a derivation or just the final formula.
import numpy


# This function finds the sample mean for one feature of a
# group of samples using maximum likelihood estimation.
# n is an integer number of samples. x is an n x 1 matrix of samples.
def find_sample_mean(n, x):
    sample_mean = numpy.zeros((1, len(x[0])))  # 1 x number of features vector of 0s

    # Iterate over all samples.
    for k in range(0, n):
        sample_mean = numpy.add(sample_mean, x[k])  # add across all features in a sample

    sample_mean = numpy.multiply(sample_mean, 1.0 / n)  # element-wise multiply
    return numpy.transpose(sample_mean)  # number of features x 1


def find_sample_variance(n, x, sample_mean):
    sample_variance = numpy.zeros((len(x[0]), len(x[0])))  # number of features x number of features vector of 0s
    sample_mean = numpy.transpose(sample_mean)  # transpose sample mean for operations

    for k in range(0, n):
        difference = numpy.subtract(x[k], sample_mean)  # difference across all features
        sample_variance = numpy.add(sample_variance, numpy.dot(numpy.transpose(difference), difference))

    sample_variance = numpy.multiply(sample_variance, 1.0 / n)  # element-wise multiply

    return sample_variance
