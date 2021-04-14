import numpy
import math


# Compute the likelihood ratio decision function
# using gaussian probability distribution function
def gaussian_likelihood(x, mu0, mu1, var0, var1, p0=0.5, p1=0.5):
    pdf1 = gaussian_pdf(x, mu0, var0)
    pdf2 = gaussian_pdf(x, mu1, var1)

    return (pdf1 / pdf2) - (p1 / p0) > 0


# Compute the probability at a certain variable for the
# gaussian probability distribution function
def gaussian_pdf(x, mu, var):
    x = numpy.mat(x)
    x = numpy.transpose(x)
    inv_var = numpy.linalg.inv(var)

    diff = x - mu
    diff_t = numpy.transpose(diff)
    exp = -0.5 * numpy.dot(numpy.dot(diff_t, inv_var), diff)
    prob = math.exp(exp)
    prob /= 2 * numpy.pi * math.sqrt(numpy.linalg.det(var))
    return prob


# Compute the likelihood ratio decision function
# using bimodal gaussian probability distribution function
def bimodal_gaussian_likelihood(x, mu0_1, mu1_1, var0_1, var1_1, a0_1, a1_1, mu0_2, mu1_2, var0_2, var1_2, a0_2, a1_2, p0=0.5, p1=0.5):
    pdf0 = bimodal_gaussian_pdf(x, mu0_1, mu1_1, var0_1, var1_1, a0_1, a1_1)
    pdf1 = bimodal_gaussian_pdf(x, mu0_2, mu1_2, var0_2, var1_2, a0_2, a1_2)

    return (pdf0 / pdf1) - (p1 / p0) > 0


# Compute the probability at a certain variable for
# the bimodal gaussian probability distribution function
def bimodal_gaussian_pdf(x, mu0, mu1, var0, var1, a0, a1):
    px0 = bimodal_gaussian_px(x, mu0, var0, a0)  # variables for peak 1
    px1 = bimodal_gaussian_px(x, mu1, var1, a1)  # variables for peak 2
    return px0 + px1


# Compute the probability at a certain variable
# for one of two modes for a bimodal gaussian
# probability distribution function
def bimodal_gaussian_px(x, mu, var, a):
    x = numpy.mat(x)
    x = numpy.transpose(x)
    inv_var = numpy.linalg.inv(var)

    diff = numpy.subtract(x, mu)
    exp = -0.5 * numpy.transpose(diff) * inv_var * diff
    px = math.exp(exp)
    px *= a / (2 * numpy.pi * math.sqrt(numpy.linalg.det(var)))
    return px
