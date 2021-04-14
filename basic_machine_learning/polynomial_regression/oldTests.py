all_linear_rmse = 0
all_linear_r2 = 0
all_but_linear_rmse = [0] * NUMBER_OF_FEATURES
all_but_linear_r2 = [0] * NUMBER_OF_FEATURES
one_linear_rmse = [0] * NUMBER_OF_FEATURES
one_linear_r2 = [0] * NUMBER_OF_FEATURES

all_quadratic_rmse = 0
all_quadratic_r2 = 0
all_but_quadratic_rmse = [0] * NUMBER_OF_FEATURES
all_but_quadratic_r2 = [0] * NUMBER_OF_FEATURES
one_quadratic_rmse = [0] * NUMBER_OF_FEATURES
one_quadratic_r2 = [0] * NUMBER_OF_FEATURES

all_cubic_rmse = 0
all_cubic_r2 = 0
all_but_cubic_rmse = [0] * NUMBER_OF_FEATURES
all_but_cubic_r2 = [0] * NUMBER_OF_FEATURES
one_cubic_rmse = [0] * NUMBER_OF_FEATURES
one_cubic_r2 = [0] * NUMBER_OF_FEATURES

# split data indices into NUMBER_OF_BATCHES batches
training_batches, testing_batches = batch(NUMBER_OF_BATCHES, len(data.data))

for training_indices, testing_indices in zip(training_batches, testing_batches):
    # extract training set
    training = Data()
    for index in training_indices:
        training.features.append(data.features[index])
        training.r.append(data.r[index])

    # extract testing set
    testing = Data()
    for index in testing_indices:
        testing.features.append(data.features[index])
        testing.r.append(data.r[index])

    # All
    polynomial = PolynomialRegression(training, testing)
    polynomial.train(1)
    real, prediction = polynomial.test(1)
    all_linear_rmse += rmse(real, prediction)
    all_linear_r2 += r_squared(real, prediction)

    polynomial = PolynomialRegression(training, testing)
    polynomial.train(2)
    real, prediction = polynomial.test(2)
    all_quadratic_rmse += rmse(real, prediction)
    all_quadratic_r2 += r_squared(real, prediction)

    polynomial = PolynomialRegression(training, testing)
    polynomial.train(3)
    real, prediction = polynomial.test(3)
    all_cubic_rmse += rmse(real, prediction)
    all_cubic_r2 += r_squared(real, prediction)

    # One/all but one
    for i in range(NUMBER_OF_FEATURES):

        # Extract just one feature
        training = Data()
        for index in training_indices:
            training.features.append([data.features[index][i]])
            training.r.append(data.r[index])

        testing = Data()
        for index in testing_indices:
            testing.features.append([data.features[index][i]])
            testing.r.append(data.r[index])

        # One
        polynomial = PolynomialRegression(training, testing)
        polynomial.train(1)
        real, prediction = polynomial.test(1)
        one_linear_rmse[i] += rmse(real, prediction)
        one_linear_r2[i] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(2)
        real, prediction = polynomial.test(2)
        one_quadratic_rmse[i] += rmse(real, prediction)
        one_quadratic_r2[i] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(3)
        real, prediction = polynomial.test(3)
        one_cubic_rmse[i] += rmse(real, prediction)
        one_cubic_r2[i] += r_squared(real, prediction)

        # Extract all but one feature
        training = Data()
        for index in training_indices:
            training.features.append(np.concatenate((data.features[index][:i], data.features[index][i + 1:])))
            training.r.append(data.r[index])

        testing = Data()
        for index in testing_indices:
            testing.features.append(np.concatenate((data.features[index][:i], data.features[index][i + 1:])))
            testing.r.append(data.r[index])

        # All but one
        polynomial = PolynomialRegression(training, testing)
        polynomial.train(1)
        real, prediction = polynomial.test(1)
        all_but_linear_rmse[i] += rmse(real, prediction)
        all_but_linear_r2[i] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(2)
        real, prediction = polynomial.test(2)
        all_but_quadratic_rmse[i] += rmse(real, prediction)
        all_but_quadratic_r2[i] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(3)
        real, prediction = polynomial.test(3)
        all_but_cubic_rmse[i] += rmse(real, prediction)
        all_but_cubic_r2[i] += r_squared(real, prediction)

print("All")
# print(all_linear_rmse / NUMBER_OF_BATCHES)
# print(all_linear_r2 / NUMBER_OF_BATCHES)
# print(all_quadratic_rmse / NUMBER_OF_BATCHES)
# print(all_quadratic_r2 / NUMBER_OF_BATCHES)
# print(all_cubic_rmse / NUMBER_OF_BATCHES)
# print(all_cubic_r2 / NUMBER_OF_BATCHES)
print("")

print("All but one")
for i in range(NUMBER_OF_FEATURES):
    print(i)
    # print(all_but_linear_rmse[i] / NUMBER_OF_BATCHES)
    print(all_linear_r2 / NUMBER_OF_BATCHES - all_but_linear_r2[i] / NUMBER_OF_BATCHES)
    # print(all_but_quadratic_rmse[i] / NUMBER_OF_BATCHES)
    print(all_quadratic_r2 / NUMBER_OF_BATCHES - all_but_quadratic_r2[i] / NUMBER_OF_BATCHES)
    # print(all_but_cubic_rmse[i] / NUMBER_OF_BATCHES)
    print(all_cubic_r2 / NUMBER_OF_BATCHES - all_but_cubic_r2[i] / NUMBER_OF_BATCHES)
    print("")

print("One")
for i in range(NUMBER_OF_FEATURES):
    print(i)
    # print(one_linear_rmse[i] / NUMBER_OF_BATCHES)
    print(one_linear_r2[i] / NUMBER_OF_BATCHES)
    # print(one_quadratic_rmse[i] / NUMBER_OF_BATCHES)
    print(one_quadratic_r2[i] / NUMBER_OF_BATCHES)
    # print(one_cubic_rmse[i] / NUMBER_OF_BATCHES)
    print(one_cubic_r2[i] / NUMBER_OF_BATCHES)
    print("")


#####################

def main():
    data = AutoData("auto-mpg.data", NUMBER_OF_FEATURES)
    data.normalize()
    data.pca_reduce(7)
    #data.stats()

    # split data indices into NUMBER_OF_BATCHES batches
    training_batches, testing_batches = batch(NUMBER_OF_BATCHES, len(data.data))

    for training_indices, testing_indices in zip(training_batches, testing_batches):

        labels = ["Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]
        for i in range(0, 7):

            # extract training set
            training = Data()
            for index in training_indices:
                training.features.append(data.features[index])
                training.r.append(data.r[index])

            # extract testing set
            testing = Data()
            for index in testing_indices:
                testing.features.append(data.features[index])
                testing.r.append(data.r[index])

            # All
            polynomial = PolynomialRegression(training, testing)
            polynomial.train(1)
            derp = np.array(polynomial.testing.features)
            for j in range(0, 7):
                if j is not i:
                    derp[:, j] = 0

            polynomial.testing.features = derp
            real, prediction = polynomial.test(1)

            plt.clf()
            plt.style.use('fivethirtyeight')
            plt.suptitle("Comparison of {} Regression vs. Real Values".format(labels[i]), fontsize=16)
            plt.subplot(121)
            plt.xlabel(labels[i])
            plt.ylabel("MPG")
            ax = plt.gca()
            ax.set_facecolor('white')
            plt.scatter(np.array(polynomial.testing.features)[:, i], real)

            plt.subplot(122)
            plt.xlabel(labels[i])
            ax = plt.gca()
            ax.set_facecolor('white')
            plt.plot(np.array(polynomial.testing.features)[:, i], prediction)
            plt.tight_layout()
            plt.subplots_adjust(top=0.8)
            plt.savefig('linear {}.png'.format(labels[i]), facecolor='white')

######

# means
    linear_means = [0] * 9
    quadratic_means = [0] * 9
    cubic_means = [0] * 9

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

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(1)
        real, prediction = polynomial.test(1)
        linear_means[0] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(2)
        real, prediction = polynomial.test(2)
        quadratic_means[0] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(3)
        real, prediction = polynomial.test(3)
        cubic_means[0] += r_squared(real, prediction)

        # Normalized
        data.normalize()
        training = Data()
        for index in training_indices:
            training.features.append(data.normalized_features[index])
            training.r.append(data.r[index])

        # extract testing set
        testing = Data()
        for index in testing_indices:
            testing.features.append(data.normalized_features[index])
            testing.r.append(data.r[index])

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(1)
        real, prediction = polynomial.test(1)
        linear_means[1] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(2)
        real, prediction = polynomial.test(2)
        quadratic_means[1] += r_squared(real, prediction)

        polynomial = PolynomialRegression(training, testing)
        polynomial.train(3)
        real, prediction = polynomial.test(3)
        cubic_means[1] += r_squared(real, prediction)

        # PCA Reduced
        for i in range(0, 7):
            start = time.time()
            data.pca_reduce(i+1)
            training = Data()
            for index in training_indices:
                training.features.append(data.reduced_features[index])
                training.r.append(data.r[index])

            # extract testing set
            testing = Data()
            for index in testing_indices:
                testing.features.append(data.reduced_features[index])
                testing.r.append(data.r[index])

            polynomial = PolynomialRegression(training, testing)
            polynomial.train(1)
            real, prediction = polynomial.test(1)
            linear_means[i+2] += r_squared(real, prediction)

            polynomial = PolynomialRegression(training, testing)
            polynomial.train(2)
            real, prediction = polynomial.test(2)
            quadratic_means[i + 2] += r_squared(real, prediction)

            polynomial = PolynomialRegression(training, testing)
            polynomial.train(3)
            real, prediction = polynomial.test(3)
            cubic_means[i + 2] += r_squared(real, prediction)
            end = time.time()
            print(i+1, end - start)

    for i, (l_mean, q_mean, c_mean) in enumerate(zip(linear_means, quadratic_means, cubic_means)):
        print(i)
        print(l_mean / (NUMBER_OF_BATCHES * 1.0))
        print(q_mean / (NUMBER_OF_BATCHES * 1.0))
        print(c_mean / (NUMBER_OF_BATCHES * 1.0))
        print("")
