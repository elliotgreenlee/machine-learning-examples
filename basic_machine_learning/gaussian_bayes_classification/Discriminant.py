import math
import numpy


# Calculate the decision rule for discriminant case 1
# x is the sample. mu0 and mu1 are the means for class 0 and 1
# var0 and var1 are the covariance matrices for class 0 and 1
# p0 and p1 are the prior probabilities for class 0 and 1
def discriminant_case1(x, mu0, mu1, var0, var1, p0=0.5, p1=0.5):

    # Calculate variance from the covariance matrices
    var = var0[0, 0] + var0[0, 1] + var1[0, 0] + var1[0, 1]
    var += var0[1, 0] + var0[1, 1] + var1[1, 0] + var1[1, 1]
    var /= 8.0

    g1 = g_case_1(x, mu0, var, p0)
    g2 = g_case_1(x, mu1, var, p1)

    return g1 - g2 > 0


# Calculate the discriminant for case 1
# x is the sample. mu is the mean. var is the
# covariance matrix. p is the prior probability
def g_case_1(x, mu, var, p):
    x = numpy.mat(x)
    x = numpy.transpose(x)

    diff = numpy.subtract(x, mu)

    g = numpy.dot(numpy.transpose(diff), diff)
    g /= -2.0 * var
    g += math.log(p)

    return g


# Calculate the decision rule for discriminant case 2
# x is the sample. mu0 and mu1 are the means for class 0 and 1
# var0 and var1 are the covariance matrices for class 0 and 1
# p0 and p1 are the prior probabilities for class 0 and 1
def discriminant_case2(x, mu0, mu1, var0, var1, p0=0.5, p1=0.5):

    # Calculate covariance matrix from covariance matrices
    var = numpy.add(var0, var1)
    var = numpy.divide(var, 2)

    inv_var = numpy.linalg.inv(var)

    g1 = g_case_2(x, mu0, inv_var, p0)
    g2 = g_case_2(x, mu1, inv_var, p1)

    return g1 - g2 > 0


# Calculate the discriminant for case 2
# x is the sample. mu is the mean. inv_var is the
# inverse of the covariance matrix.
# p is the prior probability
def g_case_2(x, mu, inv_var, p):
    x = numpy.mat(x)
    x = numpy.transpose(x)

    diff = numpy.subtract(x, mu)
    diff_transpose = numpy.transpose(diff)

    g = numpy.dot(numpy.dot(diff_transpose, inv_var), diff)
    g *= -0.5
    g += math.log(p)

    return g


# Calculate the decision rule for discriminant case 3
# x is the sample. mu0 and mu1 are the means for class 0 and 1
# var0 and var1 are the covariance matrices for class 0 and 1
# p0 and p1 are the prior probabilities for class 0 and 1
def discriminant_case3(x, mu0, mu1, var0, var1, p0=0.5, p1=0.5):
    inv_var0 = numpy.linalg.inv(var0)
    inv_var1 = numpy.linalg.inv(var1)

    g1 = g_case_3(x, mu0, inv_var0, var0, p0)
    g2 = g_case_3(x, mu1, inv_var1, var1, p1)

    return g1 - g2 > 0


# Calculate the discriminant for case 3
# x is the sample. mu is the mean. inv_var is the
# inverse of the covariance matrix.
# p is the prior probability
def g_case_3(x, mu, inv_var, var, p):
    x = numpy.mat(x)
    x = numpy.transpose(x)

    x_t = numpy.transpose(x)

    term1 = -0.5 * numpy.dot(numpy.dot(x_t, inv_var), x)

    term2 = numpy.dot(numpy.transpose(numpy.dot(inv_var, mu)), x)

    mu_t = numpy.transpose(mu)
    term3 = -0.5 * numpy.dot(numpy.dot(mu_t, inv_var), mu)
    term3 -= 0.5 * math.log(numpy.linalg.det(var))
    term3 += math.log(p)

    g = term1 + term2 + term3

    return g

