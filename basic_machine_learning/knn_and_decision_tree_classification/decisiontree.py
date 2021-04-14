"""
Elliot Greenlee
528 Project 3
November 6, 2017
"""

import math
import pandas as pd


def entropy(p1, p2):
    return -1.0 * p1 * math.log(p1, 2) - p2 * math.log(p2, 2)


def gini_index(p1, p2):
    return 2.0 * p1 * p2


def misclassification_error(p1, p2):
    return 1 - max(p1, p2)


class Node:
    def __init__(self, attribute=None, threshold=None, left_X=None, left_y=None, right_X=None, right_y=None):
        self.terminal = False
        self.prediction = None

        self.left_node = None
        self.right_node = None

        self.left_X = left_X
        self.left_y = left_y
        self.right_X = right_X
        self.right_y = right_y

        self.attribute = attribute
        self.threshold = threshold
        return


class DecisionTree:
    def __init__(self, X_train, y_train, max_depth=None, impurity_threshold=None, impurity_function=entropy):
        self.X_train = X_train
        self.y_train = y_train

        self.max_depth = max_depth
        self.impurity_threshold = impurity_threshold
        self.impurity_function = impurity_function

        self.root = None

        if max_depth is not None and impurity_threshold is not None:
            print("Choose either impurity threshold or max depth as a limiter")
            return

        if impurity_threshold is not None:
            self.max_depth = 1000000000

        if max_depth is not None:
            self.impurity_threshold = -100000000000

        self.build_tree()
        return

    def build_tree(self):
        self.root = self.create_split(self.X_train, self.y_train)
        self.root = self.split(self.root, 1)
        return

    def impurity(self, left_y, right_y):
        if left_y.empty or right_y.empty:
            return 1.0

        # Left
        left_counts = left_y['Class'].value_counts()

        if 2.0 in left_counts.index:
            benign_count = left_counts.loc[2.0]
        else:
            benign_count = 0

        if 4.0 in left_counts.index:
            malignant_count = left_counts.loc[4.0]
        else:
            malignant_count = 0

        left_p = (1.0 * benign_count) / (malignant_count + benign_count)
        left_impurity = self.impurity_f(left_p)

        # Right
        right_counts = right_y['Class'].value_counts()

        if 2.0 in right_counts.index:
            benign_count = right_counts.loc[2.0]
        else:
            benign_count = 0

        if 4.0 in right_counts.index:
            malignant_count = right_counts.loc[4.0]
        else:
            malignant_count = 0

        right_p = (1.0 * benign_count) / (malignant_count + benign_count)
        right_impurity = self.impurity_f(right_p)

        # Weight
        total = len(right_y.index) + len(left_y.index)
        right_weight = (1.0 * len(right_y.index)) / total
        left_weight = (1.0 * len(left_y.index)) / total

        return (left_impurity * left_weight) + (right_impurity * right_weight)

    # impurity measures
    def impurity_f(self, p):
        if p == 0 or p == 1:
            return 0
        else:
            return self.impurity_function(p, 1 - p)

    def create_split(self, X, y):
        min_impurity = 1.0
        threshold_attribute = None
        threshold = None
        split_left_X = None
        split_left_y = None
        split_right_X = None
        split_right_y = None

        # for each possible attribute in X
        for attribute in list(X):
            unique_values = X[attribute].unique()
            # for each row in X
            for unique_value in unique_values:
                # what are the results of this split?
                left_X, left_y, right_X, right_y = self.split_results(X, y, attribute, unique_value)

                # compute impurity for this split
                total_impurity = self.impurity(left_y, right_y)

                # if this generates the lowest impurity so far, store results
                if total_impurity <= min_impurity:
                    min_impurity = total_impurity
                    threshold_attribute = attribute
                    threshold = unique_value
                    split_left_X = left_X
                    split_left_y = left_y
                    split_right_X = right_X
                    split_right_y = right_y

        # return split with lowest impurity
        return Node(threshold_attribute, threshold, split_left_X, split_left_y, split_right_X, split_right_y)

    @staticmethod
    def split_results(X, y, attribute, threshold):

        lX = []
        ly = []
        rX = []
        ry = []

        # for rows in X and y
        for (index_X, row_X), (index_y, row_y) in zip(X.iterrows(), y.iterrows()):

            # if row value is less than threshold
            if row_X[attribute] < threshold:
                lX.append(row_X)  # append row_X to left_X
                ly.append(row_y)  # append row_y to left_y
            else:
                rX.append(row_X)  # append row_X to right_X
                ry.append(row_y)  # append row_y to left_y

        # initialize left and right X and y dataframes
        left_X = pd.DataFrame(lX, columns=list(X))
        left_y = pd.DataFrame(ly, columns=list(y))
        right_X = pd.DataFrame(rX, columns=list(X))
        right_y = pd.DataFrame(ry, columns=list(y))

        return left_X, left_y, right_X, right_y

    def split(self, node, depth):
        # TODO: maybe remove dataset from node and store in separate varibles

        # Split fully one way or the other
        if node.left_y.empty or node.right_y.empty:

            node.terminal = True
            if node.left_y.empty:
                node.prediction = node.right_y['Class'].mode()[0]
            if node.right_y.empty:
                node.prediction = node.left_y['Class'].mode()[0]
            return node

        # check depth
        if depth >= self.max_depth:
            node.left_node = Node()
            node.left_node.terminal = True
            node.left_node.prediction = node.left_y['Class'].mode()[0]

            node.right_node = Node()
            node.right_node.terminal = True
            node.right_node.prediction = node.right_y['Class'].mode()[0]

            return node

        # check impurity threshold
        total_impurity = self.impurity(node.left_y, node.right_y)
        if total_impurity <= self.impurity_threshold:
            node.left_node = Node()
            node.left_node.terminal = True
            node.left_node.prediction = node.left_y['Class'].mode()[0]

            node.right_node = Node()
            node.right_node.terminal = True
            node.right_node.prediction = node.right_y['Class'].mode()[0]

            return node

        # left child
        node.left_node = self.create_split(node.left_X, node.left_y)
        node.left_node = self.split(node.left_node, depth + 1)

        # right child
        node.right_node = self.create_split(node.right_X, node.right_y)
        node.right_node = self.split(node.right_node, depth + 1)

        return node

    def print_node(self, node, depth):
        print(("\t" * depth) + "Depth: {}".format(depth))
        if node.terminal:
            print(("\t" * depth) + "{}".format(node.prediction))
        else:
            print(("\t" * depth) + "{}".format(node.attribute))
            print(("\t" * depth) + "{}".format(node.threshold))
            print(("\t" * depth) + "left")
            self.print_node(node.left_node, depth + 1)
            print(("\t" * depth) + "right")
            self.print_node(node.right_node, depth + 1)

        return

    def print_tree(self):
        print("Tree")
        self.print_node(self.root, 1)
        return

    def predict_node(self, node, x):
        if node.terminal:
            return node.prediction
        else:
            if x[node.attribute] < node.threshold:
                return self.predict_node(node.left_node, x)
            else:
                return self.predict_node(node.right_node, x)

    def predict_tree(self, x):
        return self.predict_node(self.root, x)
