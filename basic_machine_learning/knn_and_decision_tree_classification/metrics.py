"""
Elliot Greenlee
528 Project 3
November 5, 2017
"""

from sklearn.metrics import confusion_matrix


def metrics(y_true, y_predicted):
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(y_true, y_predicted).ravel()

    # Confusion matrix
    print("Confusion Matrix")
    print("\t\t\t|Predicted Class\t\t|")
    print("True Class\t|benign\t\t|malignant\t|")
    print("benign\t\t|{:03d}\t\t|{:03d}\t\t|".format(true_negatives, false_positives))
    print("malignant\t|{:03d}\t\t|{:03d}\t\t|".format(false_negatives, true_positives))
    print("")

    # Accuracy
    accuracy = (1.0 * (true_negatives + true_positives)) / (true_negatives + true_positives + false_negatives + false_positives)
    print("Accuracy")
    print(accuracy)
    print("")

    # True Positive Rate
    true_positive_rate = (1.0 * true_positives) / (true_positives + false_negatives)
    print("True Positive Rate, Recall, Sensitivity")
    print(true_positive_rate)
    print("")

    # Positive Predictive Value
    positive_predictive_value = (1.0 * true_positives) / (true_positives + false_positives)
    print("Positive Predictive Value, Precision")
    print(positive_predictive_value)
    print("")

    # True Negative Rate
    true_negative_rate = (1.0 * true_negatives) / (true_negatives + false_positives)
    print("True Negative Rate, Specificity")
    print(true_negative_rate)
    print("")

    # F Score
    f_score = (1.0 * (positive_predictive_value * true_positive_rate)) / (positive_predictive_value + true_positive_rate)
    print("F Score")
    print(f_score)
    print("")
