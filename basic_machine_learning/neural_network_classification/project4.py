"""
Elliot Greenlee
528 Project 4
November 15, 2017
"""
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import scale
import numpy as np
import time
import matplotlib.pyplot as plt

from spam import Spam
from neural_network import NeuralNetwork


def confuse(y_true, y_predicted):

    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_true, y_predicted).ravel()

    # Confusion matrix
    print("Confusion Matrix")
    print("\t\t\t|Predicted Class\t\t|")
    print("True Class\t|not spam\t\t|spam\t|")
    print("not spam\t\t|{:03d}\t\t|{:03d}\t\t|".format(true_negatives, false_positives))
    print("spam\t|{:03d}\t\t|{:03d}\t\t|".format(false_negatives, true_positives))
    print("")


# Set random seed
np.random.seed(93874)

# Load the data
spam = Spam()

# Standardize the data
spam.x = scale(spam.x)

# Split the data for evaluation
spam.train_test_split(0.2, True)
spam.k_fold(5)

"""
# Cross validation
# Neural Network parameters
structure = [spam.number_of_features, 50, 1]
learning_rate = 0.01
weight_variances = [5.0, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
#stopping_condition = 'error_change_threshold'
#stopping_condition = 'maximum_epochs'
stopping_condition = 'error_threshold'
error_threshold = 0.35
error_change_threshold = 0.0001
maximum_epochs = 500
output_layer = 'logistic_sigmoid'

for weight_variance in weight_variances:
    print("weight_variance: {}".format(weight_variances))
    fold = 0
    validation_accuracy = 0
    epochs = 0
    train_time = 0
    for training_index, validation_index in spam.k_fold_splits:
        training_x = spam.training_x[training_index]
        training_y = spam.training_y[training_index]
        validation_x = spam.training_x[validation_index]
        validation_y = spam.training_y[validation_index]

        neural_network = NeuralNetwork(structure, output_layer=output_layer, weight_variance=weight_variance)

        start_time = time.time()
        epochs += neural_network.train(training_x, training_y, validation_x, validation_y,
                                       learning_rate,
                                       stopping_condition=stopping_condition,
                                       error_change_threshold=error_change_threshold,
                                       maximum_epochs=maximum_epochs,
                                       error_threshold=error_threshold)
        end_time = time.time()
        train_time += end_time - start_time

        predicted_y = neural_network.predict(validation_x)
        validation_accuracy += accuracy_score(validation_y, predicted_y)

        fold += 1

    print("Validation Accuracy: {}".format(validation_accuracy / fold))
    print("Epochs: {}".format(epochs / fold))
    print("Train Time: {}".format(train_time / fold))

exit(1)
"""

# Neural Network parameters
structure = [spam.number_of_features, 200, 200, 1]
learning_rate = 0.01
weight_variance = 0.2
#stopping_condition = 'error_change_threshold'
stopping_condition = 'maximum_epochs'
error_change_threshold = 0.0001
maximum_epochs = 120
output_layer = 'logistic_sigmoid'

# Training
print("Building Network")
neural_network = NeuralNetwork(structure, output_layer=output_layer, weight_variance=weight_variance)

print("Training Network")
epochs, training_accs, validation_accs = neural_network.train(
    spam.training_x, spam.training_y, spam.testing_x, spam.testing_y,
    learning_rate,
    stopping_condition=stopping_condition,
    error_change_threshold=error_change_threshold, maximum_epochs=maximum_epochs)

# Testing
predicted_y = neural_network.predict(spam.testing_x)
testing_accuracy = accuracy_score(spam.testing_y, predicted_y)
print("Testing Accuracy: {}".format(testing_accuracy))
confuse(spam.testing_y, predicted_y)

# Graph
ep = list(range(epochs-1))
plt.plot(ep, validation_accs)
plt.plot(ep, training_accs)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
