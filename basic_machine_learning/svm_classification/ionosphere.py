"""
Elliot Greenlee
2017-12-1
UTK COSC 528 Project 5
"""

import os
import pandas as pd

from data_helper.supervised_data import SupervisedData


class Ionosphere(SupervisedData):
    def __init__(self):
        SupervisedData.__init__(self)

    def load(self):
        """
        :return: x, y
        x is an mxn numpy array of the data where m is the number of samples and n is the number of features.
        y is an mx1 numpy array of the classes where m is the number of samples.
        Each value in y is an integer class label indexed from 0.
        """

        data_directory = "data/ionosphere"
        data_filename = "ionosphere.data"
        data_path = os.path.join(data_directory, data_filename)

        df = pd.read_csv(data_path, header=None)  # Load all data

        x = df.as_matrix([i for i in range(len(df.columns)-1)])  # Load x

        y = df.as_matrix([len(df.columns)-1])  # Load y
        y = (y == 'g').astype(int)  # convert g to 1 and b to 0

        return x, y

    def label_features(self):
        """
        :return: feature_labels
        feature_labels is an n length list of strings of the label for each feature, where n is the number of features.
        """

        feature_labels = [
            "pulse1 attribute1",
            "pulse1 attribute2",
            "pulse2 attribute1",
            "pulse2 attribute2",
            "pulse3 attribute1",
            "pulse3 attribute2",
            "pulse4 attribute1",
            "pulse4 attribute2",
            "pulse5 attribute1",
            "pulse5 attribute2",
            "pulse6 attribute1",
            "pulse6 attribute2",
            "pulse7 attribute1",
            "pulse7 attribute2",
            "pulse8 attribute1",
            "pulse8 attribute2",
            "pulse9 attribute1",
            "pulse9 attribute2",
            "pulse10 attribute1",
            "pulse10 attribute2",
            "pulse11 attribute1",
            "pulse11 attribute2",
            "pulse12 attribute1",
            "pulse12 attribute2",
            "pulse13 attribute1",
            "pulse13 attribute2",
            "pulse14 attribute1",
            "pulse14 attribute2",
            "pulse15 attribute1",
            "pulse15 attribute2",
            "pulse16 attribute1",
            "pulse16 attribute2",
            "pulse17 attribute1",
            "pulse17 attribute2"
        ]

        return feature_labels

    def label_classes(self):
        """
        :return: class_labels
        class_labels is an l length list of strings of the label for each class, where l is the number of classes.
        """

        class_labels = [
            "bad",
            "good"
        ]

        return class_labels
