"""
Elliot Greenlee
528 Project 3
November 5, 2017
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from stats import stats
from knn import knn
from metrics import metrics
from decisiontree import DecisionTree, entropy, gini_index, misclassification_error


data_file = "data/breast-cancer-wisconsin.data"
original_data = pd.read_csv(data_file)  # 699 rows x 11 columns, Bare Nuclei missing 16 values
full_data = original_data.dropna(axis=0, how='any')  # 683 rows x 11 columns, no missing values
data = full_data.drop('Sample Code Number', axis=1)  # 683 rows x 10 columns
data['Bare Nuclei'] = data['Bare Nuclei'].astype(np.int64)

# Statistics and Data exploration
# stats(data)


X = data.drop('Class', axis=1)
y = data['Class'].to_frame(name='Class')

# Split into cross validation and testing set
X_cross_validate, X_test, y_cross_validate, y_test = train_test_split(X, y, test_size=0.50, random_state=736)

# Split cross validations set into training and testing
kf = KFold(n_splits=10, shuffle=False)
kf.get_n_splits(X_cross_validate)


"""
# kNN Testing
ks = [2, 3, 4, 5, 6, 7, 8, 16, 32]
for k in ks:
    print("k = {}".format(k))

    # Split data for cross validation and iterate over it
    y_validate_full = []
    y_predicted_full = []
    for train_index, validate_index in kf.split(X_cross_validate):
        # print("Train:", train_index, "Validate:", validate_index)
        X_train, X_validate = X_cross_validate.iloc[train_index], X_cross_validate.iloc[validate_index]
        y_train, y_validate = y_cross_validate.iloc[train_index], y_cross_validate.iloc[validate_index]

        # Predict each class using kNN
        y_predicted = []
        for x_index, x in X_validate.iterrows():
            y_prediction = knn(k, x, X_train, y_train)
            y_predicted.append(y_prediction)

        y_predicted = pd.DataFrame(np.array(y_predicted).reshape((len(y_predicted), 1)), columns=['Class'])

        # append results for all cross validations splits
        y_validate_full.append(y_validate)
        y_predicted_full.append(y_predicted)

    # print metrics for this k value
    metrics(pd.concat(y_validate_full), pd.concat(y_predicted_full))
"""

# Decision Tree Testing
depths = [2, 3, 4, 5, 6, 7, 8, 16, 32]
impurity_thresholds = []
for depth in depths:
    print("depth = {}".format(depth))

    # Split data for cross validation and iterate over it
    y_validate_full = []
    y_predicted_full = []
    for train_index, validate_index in kf.split(X_cross_validate):
        print("Next Fold")
        # print("Train:", train_index, "Validate:", validate_index)
        X_train, X_validate = X_cross_validate.iloc[train_index], X_cross_validate.iloc[validate_index]
        y_train, y_validate = y_cross_validate.iloc[train_index], y_cross_validate.iloc[validate_index]

        # Build decision tree
        decision_tree = DecisionTree(X_train, y_train, max_depth=depth, impurity_function=gini_index)
        #decision_tree.print_tree()  # print tree

        # Predict each class using decision tree
        y_predicted = []
        for x_index, x in X_validate.iterrows():
            y_prediction = decision_tree.predict_tree(x)
            y_predicted.append(y_prediction)

        y_predicted = pd.DataFrame(np.array(y_predicted).reshape((len(y_predicted), 1)), columns=['Class'])

        # append results for all cross validations splits
        y_validate_full.append(y_validate)
        y_predicted_full.append(y_predicted)

    # print metrics for this depth
    metrics(pd.concat(y_validate_full), pd.concat(y_predicted_full))

exit(1)
# Dimensionality reduction testing
Xs = []
for n_components in range(1,9):
    # Reduce dimensionality by PCA
    pca = PCA(n_components=n_components)
    Xs.append(pca.fit_transform(X))

for X in Xs:
    # Split into cross validation and testing set
    X_cross_validate, X_test, y_cross_validate, y_test = train_test_split(X, y, test_size=0.30, random_state=736)

    # Split cross validations set into training and testing
    kf = KFold(n_splits=10, shuffle=False)
    kf.get_n_splits(X_cross_validate)
    for train_index, validate_index in kf.split(X_cross_validate):
        # print("Train:", train_index, "Validate:", validate_index)
        X_train, X_validate = X_cross_validate.iloc[train_index], X_cross_validate.iloc[validate_index]
        y_train, y_validate = y_cross_validate.iloc[train_index], y_cross_validate.iloc[validate_index]

        # true_negatives, false_positives, false_negatives, true_positives =
        # confusion_matrix(y_true, y_predicted).ravel()
