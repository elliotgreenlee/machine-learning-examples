"""
Elliot Greenlee
2017-12-2
UTK COSC 528 Project 5
"""

import os
import pandas as pd

from data_helper.supervised_data import SupervisedData


class VowelContext(SupervisedData):
    def __init__(self):
        SupervisedData.__init__(self)

    def load(self):
        """
        :return: x, y
        x is an mxn numpy array of the data where m is the number of samples and n is the number of features.
        y is an mx1 numpy array of the classes where m is the number of samples.
        Each value in y is an integer class label indexed from 0.
        """

        data_directory = "data/vowel-context"
        data_filename = "vowel-context.data"
        data_path = os.path.join(data_directory, data_filename)

        df = pd.read_csv(data_path, header=None, delim_whitespace=True)  # Load data

        x = df.as_matrix([i for i in range(3, len(df.columns)-1)])  # Load x

        y = df.as_matrix([len(df.columns)-1])  # Load y

        return x, y

    def label_features(self):
        """
        :return: feature_labels
        feature_labels is an n length list of strings of the label for each feature, where n is the number of features.
        """

        feature_labels = [
            "Feature0",
            "Feature1",
            "Feature2",
            "Feature3",
            "Feature4",
            "Feature5",
            "Feature6",
            "Feature7",
            "Feature8",
            "Feature9"
        ]

        return feature_labels

    def label_classes(self):
        """
        :return: class_labels
        class_labels is an l length list of strings of the label for each class, where l is the number of classes.
        """

        class_labels = [
            "hid",
            "hId",
            "hEd",
            "hAd",
            "hYd",
            "had",
            "hOd",
            "hod",
            "hUd",
            "hud",
            "hed"
        ]

        return class_labels
