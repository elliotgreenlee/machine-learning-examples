"""
Elliot Greenlee
528 Project 4
November 15, 2017
"""

import pickle
import numpy as np
import scipy
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle


def one_hot(ys):
    one_hot_y = []
    for y in ys:
        if y == 0:
            one_hot_y.append([1, 0])
        elif y == 1:
            one_hot_y.append([0, 1])
    one_hot_y = np.array(one_hot_y)
    return one_hot_y


class NeuralNetwork:
    def __init__(self, structure, output_layer, weight_variance):
        # Build graph
        self.weights = None
        self.output_layer = None
        self.d_output_layer = None
        self.build_graph(structure, output_layer, weight_variance)

    """Building the graph"""
    def build_graph(self, structure, output_layer, weight_variance):
        self.weights = []
        for layer in range(len(structure)-1):
            if layer == (len(structure) - 2):
                weight_shape = (structure[layer] + 1, structure[layer + 1])
            else:
                weight_shape = (structure[layer] + 1, structure[layer+1] + 1)
            # Initialize weights
            mean = 0
            weights = np.random.normal(mean, weight_variance, weight_shape)
            self.weights.append(weights.reshape(weight_shape))

        # Determine output layer function
        self.output_layer, self.d_output_layer = self.determine_output_layer(output_layer, structure[-1])
        return

    def determine_output_layer(self, output_layer, output_size):
        if output_layer is "linear":
            if output_size is not 1:
                print("Error: linear output layer selected with a size of {} instead of 1.\n"
                      "Choose a structure size of 1 and make sure your ground truth values are correct.")
                exit(1)
            return self.linear, self.d_linear
        elif output_layer is "logistic_sigmoid":
            if output_size is not 1:
                print("Error: logistic_sigmoid output layer selected with a size of {} instead of 1.\n"
                      "Choose a structure size of 1 and make sure your ground truth values are correct.")
                exit(1)
            return self.logistic_sigmoid, self.d_logistic_sigmoid
        elif output_layer is "softmax":
            if output_size is not 2:
                print("Error: softmax output layer selected with a size of {} instead of 1.\n"
                      "Choose a structure size of 1 and make sure your ground truth values are correct.")
                exit(1)
            return self.softmax, self.d_softmax
        else:
            print("Error: output_layer {} is not valid.".format(output_layer))
            exit(1)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def d_linear(x):
        return 1.0

    @staticmethod
    def logistic_sigmoid(x):
        return scipy.special.expit(x)

    def d_logistic_sigmoid(self, x):
        y = self.logistic_sigmoid(x)
        return y * (1.0 - y)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # subtracting by max prevents big exponents
        return e_x / e_x.sum(axis=0)

    def d_softmax(self, x):
        soft = self.softmax(x)
        s = soft.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    """Training the graph"""
    def train(self, training_x, training_y, validation_x, validation_y,
              learning_rate,
              stopping_condition, maximum_epochs=None, error_threshold=None, error_change_threshold=None):

        # Determine stopping condition
        stop, threshold = self.determine_stop(stopping_condition,
                                              maximum_epochs, error_threshold, error_change_threshold)

        if self.output_layer == self.softmax:
            t_y = one_hot(training_y)
        else:
            t_y = training_y

        # Intialize
        current_epoch = 1
        old_error = 2.0
        current_error = 1.0
        current_error_change = 1.0

        training_accuracies = []
        validation_accuracies = []

        # While the chosen stop condition has not been reached
        while not stop(threshold,
                       current_epoch=current_epoch,
                       current_error=current_error,
                       current_error_change=current_error_change):
            # Update the training weights
            batch_x, batch_y = shuffle(training_x, t_y)
            current_error = self.update_weights(batch_x, batch_y, learning_rate)

            # Evaluate the validation data
            predicted_y = self.predict(validation_x)
            validation_accuracy = accuracy_score(validation_y, predicted_y)

            # Evaluate the training data
            predicted_y = self.predict(training_x)
            training_accuracy = accuracy_score(training_y, predicted_y)

            # Calculate the error change
            current_error_change = old_error - current_error

            # Print results
            training_accuracies.append(training_accuracy)
            validation_accuracies.append(validation_accuracy)
            """
            print("Epoch: {}".format(current_epoch))
            print("\tError: {}".format(current_error))
            print("\tError Change: {}".format(current_error_change))
            print("\tTraining Accuracy: {}".format(training_accuracy))
            print("\tValidation Accuracy: {}".format(validation_accuracy))
            """

            # Setup for next epoch
            old_error = current_error
            current_epoch += 1

        return current_epoch, training_accuracies, validation_accuracies

    def determine_stop(self, stopping_condition,
                       maximum_epochs=None,
                       error_threshold=None,
                       error_change_threshold=None):
        if stopping_condition is "maximum_epochs":
            if maximum_epochs is None:
                print("Error: maximum_epochs was selected as a stopping condition "
                      "but no maximum_epochs value was given.")
                exit(1)

            return self.epochs_stop, maximum_epochs
        elif stopping_condition is "error_threshold":
            if error_threshold is None:
                print("Error: error_threshold was selected as a stopping condition "
                      "but no error_threshold value was given.")
                exit(1)

            return self.error_threshold_stop, error_threshold
        elif stopping_condition is "error_change_threshold":
            if error_change_threshold is None:
                print("Error: error_change_threshold was selected as a stopping condition "
                      "but no error_change_threshold value was given.")
                exit(1)

            return self.error_change_threshold_stop, error_change_threshold
        else:
            print("Error: stop condition {} is not valid.".format(stopping_condition))
            exit(1)

    @staticmethod
    def epochs_stop(maximum_epochs,
                    current_epoch,
                    current_error=None,
                    current_error_change=None):
        return current_epoch >= maximum_epochs

    @staticmethod
    def error_threshold_stop(error_threshold,
                             current_error,
                             current_epoch=None,
                             current_error_change=None):
        return current_error <= error_threshold

    @staticmethod
    def error_change_threshold_stop(error_change_threshold,
                                    current_error_change,
                                    current_epoch=None,
                                    current_error=None):
        return current_error_change <= error_change_threshold

    def update_weights(self, batch_x, batch_y, learning_rate):
        # append the biases
        batch_x = np.append(batch_x, np.ones((len(batch_x), 1)), axis=1)

        ys = []
        # For all inputs (update via online training)
        for x, y_ in zip(batch_x, batch_y):
            y = None

            # Forward Propagation
            x_layers = []
            # iterate through the layers
            for layer, weights in enumerate(self.weights):
                # Save x values at each layer for later
                x_layers.append(x.reshape((1, len(x))))

                # Linear combination
                z = x.dot(weights)

                # Activation function dependent on layer
                if layer == (len(self.weights) - 1):
                    y = self.output_layer(z)
                else:
                    x = self.logistic_sigmoid(z)

            # Backpropagation
            # Initialize the deltas
            delta_ws = [0] * len(self.weights)

            # Calculate delta for the output layer
            l = len(self.weights) - 1
            delta_ws[l] = np.transpose(x_layers[l]).dot(np.matrix(y - y_))
            delta_x = self.weights[l].dot(np.transpose(np.matrix(y - y_)))

            # Iterate over hidden layers
            for layer in range(len(self.weights) - 2, -1, -1):
                z = x_layers[layer].dot(self.weights[layer])
                tmp = np.multiply(delta_x, np.transpose(self.d_logistic_sigmoid(z)))
                delta_ws[layer] = np.transpose(x_layers[layer]).dot(np.transpose(tmp))
                delta_x = self.weights[layer].dot(tmp)

            # Update the weights
            for layer in range(1, len(self.weights)):
                self.weights[layer] -= learning_rate * delta_ws[layer]

            # Add the calculated y to the results
            ys.append(y)

        # Calculate the error for this batch
        ys = np.array(ys)
        error = log_loss(batch_y, ys)

        return error

    """Evaluate"""
    def predict(self, xs):
        ys = self.evaluating_propagate(xs)

        if len(ys[0]) == 1:
            # 0 if 0.5 or lower, else 1
            zeros = np.where(ys <= 0.5)
            ones = np.where(ys > 0.5)
            predicted_y = np.array(ys)
            predicted_y[zeros] = 0
            predicted_y[ones] = 1
            return predicted_y
        elif len(ys[0]) == 2:
            # 0 if [0] >= [1], else 1
            predicted_y = np.argmax(ys, axis=1)

            return predicted_y
        else:
            print("Error: ")

    def evaluating_propagate(self, xs):
        ys = []
        xs = np.append(xs, np.ones((len(xs), 1)), axis=1)
        for x in xs:
            y = None
            for i, weights in enumerate(self.weights):
                z = x.dot(weights)
                if i == (len(self.weights) - 1):
                    y = self.output_layer(z)
                else:
                    a = self.logistic_sigmoid(z)
                    x = a
            ys.append(y)

        ys = np.array(ys)
        return ys

    """Store and Load the trained weights"""
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)
            # TODO: dump biases
            # TODO: dump output layer function
        return

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f)
            # TODO: load biases
            # TODO: load output layer function
