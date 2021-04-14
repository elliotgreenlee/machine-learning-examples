from Data import *
from PolynomialRegression import PolynomialRegression

import numpy as np
import math


def rmse(real, prediction):
    error = 0
    for r, p in zip(real, prediction):
        error += math.pow(r - p, 2)

    error = math.sqrt(error / (len(real) * 1.0))
    return error


def r_squared(real, prediction):
    mean = 0
    for instance in real:
        mean += instance
    mean /= len(real) * 1.0

    ss_tot = 0
    for instance in real:
        ss_tot += math.pow(instance - mean, 2)

    ss_res = 0
    for r, p in zip(real, prediction):
        ss_res += math.pow(r - p, 2)

    r2 = 1 - (ss_res / (ss_tot * 1.0))
    return r2


def print_comparison(real, prediction):
    print("Real : Prediction")
    for r, p in zip(real, prediction):
        print("{} : {}".format(r, p))


NUMBER_OF_FEATURES = 7
NUMBER_OF_BATCHES = 5


def batch(number_of_batches, number_of_samples):

    all_indices = np.arange(0, number_of_samples)  # get all possible indices
    np.random.shuffle(all_indices)

    if number_of_batches is 1:
        TRAINING_SIZE = 292
        training = [all_indices[:TRAINING_SIZE]]
        testing = [all_indices[TRAINING_SIZE:]]
        return training, testing

    remainder = number_of_samples % number_of_batches
    samples_per_batch = int(number_of_samples / number_of_batches)

    training_batches = []
    testing_batches = []
    current_index = 0
    for i in range(number_of_batches):
        old_index = current_index
        current_index += samples_per_batch
        if i < remainder:
            current_index += 1

        train_batch = np.concatenate((all_indices[:old_index], all_indices[current_index:]))
        test_batch = all_indices[old_index:current_index]

        training_batches.append(train_batch)
        testing_batches.append(test_batch)

    return training_batches, testing_batches


def main():
    data = AutoData("auto-mpg.data", NUMBER_OF_FEATURES)
    data.normalize()
    data.pca_reduce(4)
    data.stats()

    # split data indices into NUMBER_OF_BATCHES batches
    training_batches, testing_batches = batch(NUMBER_OF_BATCHES, len(data.data))

    # for each batch
    for training_indices, testing_indices in zip(training_batches, testing_batches):

        # Nothing
        training = Data()
        for index in training_indices:
            training.features.append(data.features[index])
            training.r.append(data.r[index])

        # extract testing set
        testing = Data()
        for index in testing_indices:
            testing.features.append(data.features[index])
            testing.r.append(data.r[index])

        # Run regression
        polynomial = PolynomialRegression(training, testing)
        polynomial.train(1)
        real, prediction = polynomial.test(1)
        print(r_squared(real, prediction))

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(2)
        real, prediction = polynomial.test(2)
        print(r_squared(real, prediction))

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(3)
        real, prediction = polynomial.test(3)
        print(r_squared(real, prediction))

if __name__ == "__main__":
    main()
