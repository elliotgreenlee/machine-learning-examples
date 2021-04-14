"""
Elliot Greenlee
528 Project 3
November 5, 2017
"""

import numpy as np
from scipy.spatial import distance


def knn(k, x, X_train, y_train):
    # iterate over all training samples
    distances = []
    indices = []
    for sample_index, sample in X_train.iterrows():
        # compute distance
        indices.append(sample_index)
        distances.append(distance.euclidean(x, sample))

    # sort by minimum distance
    distances = np.array(distances)
    indices = np.array(indices)
    sorted_index = np.argsort(distances)

    # sort training data and labels by minimum distance
    y_closest = y_train.loc[indices[sorted_index]].iloc[0:k]
    X_train_closest = X_train.loc[indices[sorted_index]].iloc[0:k]

    # count each class in the closest k
    benign_count = 0
    malignant_count = 0
    for _, y in y_closest.iterrows():
        if y['Class'] == 2:
            benign_count += 1
        else:
            malignant_count += 1

    # break ties in favor of malignant
    if benign_count > malignant_count:
        y_prediction = 2
    else:
        y_prediction = 4

    return y_prediction
