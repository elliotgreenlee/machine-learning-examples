"""
Elliot Greenlee
2017-12-1
UTK COSC 528 Project 5
"""

import os
import pandas as pd

from data_helper.supervised_data import SupervisedData


class Sat(SupervisedData):
    def __init__(self):
        SupervisedData.__init__(self)

    def load(self):
        """
        :return: x, y
        x is an mxn numpy array of the data where m is the number of samples and n is the number of features.
        y is an mx1 numpy array of the classes where m is the number of samples.
        Each value in y is an integer class label indexed from 0.
        """

        data_directory = "data/sat"
        training_filename = "sat.trn"
        testing_filename = "sat.tst"
        training_path = os.path.join(data_directory, training_filename)
        testing_path = os.path.join(data_directory, testing_filename)

        training_df = pd.read_csv(training_path, header=None, delim_whitespace=True)  # Load training data
        testing_df = pd.read_csv(testing_path, header=None, delim_whitespace=True)  # Load testing data

        df = pd.concat([training_df, testing_df])

        x = df.as_matrix([i for i in range(len(df.columns)-1)])  # Load x

        y = df.as_matrix([len(df.columns)-1])  # Load y

        return x, y

    def label_features(self):
        """
        :return: feature_labels
        feature_labels is an n length list of strings of the label for each feature, where n is the number of features.
        """

        feature_labels = [
            "top left pixel spectral band 1",
            "top left pixel spectral band 2",
            "top left pixel spectral band 3",
            "top left pixel spectral band 4",
            "top middle pixel spectral band 1",
            "top middle pixel spectral band 2",
            "top middle pixel spectral band 3",
            "top middle pixel spectral band 4",
            "top right pixel spectral band 1",
            "top right pixel spectral band 2",
            "top right pixel spectral band 3",
            "top right pixel spectral band 4",
            "middle left pixel spectral band 1",
            "middle left pixel spectral band 2",
            "middle left pixel spectral band 3",
            "middle left pixel spectral band 4",
            "middle middle pixel spectral band 1",
            "middle middle pixel spectral band 2",
            "middle middle pixel spectral band 3",
            "middle middle pixel spectral band 4",
            "middle right pixel spectral band 1",
            "middle right pixel spectral band 2",
            "middle right pixel spectral band 3",
            "middle right pixel spectral band 4",
            "bottom left pixel spectral band 1",
            "bottom left pixel spectral band 2",
            "bottom left pixel spectral band 3",
            "bottom left pixel spectral band 4",
            "bottom middle pixel spectral band 1",
            "bottom middle pixel spectral band 2",
            "bottom middle pixel spectral band 3",
            "bottom middle pixel spectral band 4",
            "bottom right pixel spectral band 1",
            "bottom right pixel spectral band 2",
            "bottom right pixel spectral band 3",
            "bottom right pixel spectral band 4"
        ]

        return feature_labels

    def label_classes(self):
        """
        :return: class_labels
        class_labels is an l length list of strings of the label for each class, where l is the number of classes.
        """

        class_labels = [
            "red soil",
            "cotton crop",
            "grey soil",
            "damp grey soil",
            "soil with vegetable stubble",
            "mixture class (all types present)",
            "very damp grey soil"
        ]

        return class_labels
