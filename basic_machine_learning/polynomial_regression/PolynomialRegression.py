import numpy


class PolynomialRegression:
    def __init__(self, training, testing):
        self.training = training
        self.testing = testing

        self.w = None

    @staticmethod
    def linear(input_features, input_r):
        x = numpy.array(input_features)
        ones = numpy.ones((len(x), 1))
        x = numpy.concatenate((ones, x), 1)  # add a vector of ones in front
        r = numpy.array(input_r)
        return x, r

    @staticmethod
    def quadratic(input_features, input_r):
        pre_features = numpy.array(input_features)
        ones = numpy.ones((len(pre_features), 1))
        pre_features = numpy.concatenate((ones, pre_features), 1)  # add a vector of ones in front

        quadratic_x = []
        for instance in pre_features:
            row_features = []
            for i in range(0, len(instance)):
                for j in range(i, len(instance)):
                    row_features.append(instance[i] * instance[j])
            quadratic_x.append(row_features)

        x = numpy.array(quadratic_x)

        r = numpy.array(input_r)
        return x, r

    @staticmethod
    def cubic(input_features, input_r):
        pre_features = numpy.array(input_features)
        ones = numpy.ones((len(pre_features), 1))
        pre_features = numpy.concatenate((ones, pre_features), 1)  # add a vector of ones in front

        cubic_x = []
        for instance in pre_features:
            row_features = []
            for i in range(0, len(instance)):
                for j in range(i, len(instance)):
                    for k in range(j, len(instance)):
                        row_features.append(instance[i] * instance[j] * instance[k])
            cubic_x.append(row_features)

        x = numpy.array(cubic_x)

        r = numpy.array(input_r)
        return x, r

    def train(self, order):
        if order is 1:
            x, r = self.linear(self.training.features, self.training.r)
        elif order is 2:
            x, r = self.quadratic(self.training.features, self.training.r)
        elif order is 3:
            x, r = self.cubic(self.training.features, self.training.r)
        else:
            print("Order {} is not supported.".format(order))
            exit(1)
            return

        # W = (X^T * X)^-1 * X^T * R
        x_transpose = numpy.transpose(x)
        xtx = numpy.matmul(x_transpose, x)
        xtx1 = numpy.linalg.inv(xtx)
        xtx1_xt = numpy.matmul(xtx1, x_transpose)
        xtx1_xt_r = numpy.matmul(xtx1_xt, r)

        w = xtx1_xt_r

        self.w = w

    def test(self, order):
        if order is 1:
            x, r = self.linear(self.testing.features, self.testing.r)
        elif order is 2:
            x, r = self.quadratic(self.testing.features, self.testing.r)
        elif order is 3:
            x, r = self.cubic(self.testing.features, self.testing.r)
        else:
            print("Order {} is not supported.".format(order))
            exit(1)
            return

        w_transpose = numpy.transpose(self.w)

        real = []
        prediction = []
        for instance, r in zip(x, r):
            p = numpy.matmul(instance, w_transpose)
            prediction.append(p)
            real.append(r)

        return real, prediction
