from data_helper.supervised_data import SupervisedData

import os
import pandas as pd


class Spam(SupervisedData):
    def __init__(self):
        SupervisedData.__init__(self)

    def load(self):
        data_directory = "data"
        data_file = "spambase.data"
        data_path = os.path.join(data_directory, data_file)

        df = pd.read_csv(data_path, header=None)

        x = df.as_matrix([i for i in range(len(df.columns)-1)])
        y = df.as_matrix([len(df.columns)-1])

        return x, y

    def label_features(self):
        feature_labels = [
            "word_freq_make",
            "word_freq_address",
            "word_freq_all",
            "word_freq_3d",
            "word_freq_our",
            "word_freq_over",
            "word_freq_remove",
            "word_freq_internet",
            "word_freq_order",
            "word_freq_mail",
            "word_freq_receive",
            "word_freq_will",
            "word_freq_people",
            "word_freq_report",
            "word_freq_addresses",
            "word_freq_free",
            "word_freq_business",
            "word_freq_email",
            "word_freq_you",
            "word_freq_credit",
            "word_freq_your",
            "word_freq_font",
            "word_freq_000",
            "word_freq_money",
            "word_freq_hp",
            "word_freq_hpl",
            "word_freq_george",
            "word_freq_650",
            "word_freq_lab",
            "word_freq_labs",
            "word_freq_telnet",
            "word_freq_857",
            "word_freq_data",
            "word_freq_415",
            "word_freq_85",
            "word_freq_technology",
            "word_freq_1999",
            "word_freq_parts",
            "word_freq_pm",
            "word_freq_direct",
            "word_freq_cs",
            "word_freq_meeting",
            "word_freq_original",
            "word_freq_project",
            "word_freq_re",
            "word_freq_edu",
            "word_freq_table",
            "word_freq_conference",
            "char_freq_;",
            "char_freq_(",
            "char_freq_[",
            "char_freq_!",
            "char_freq_$",
            "char_freq_#",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total"
        ]

        return feature_labels

    def label_classes(self):
        class_labels = [
            "Not Spam",
            "Spam"
        ]

        return class_labels

spam = Spam()

"""
print(spam.x)
print(spam.y)
print(spam.feature_labels)
print(spam.class_labels)
print(spam.dataframe)
"""

"""
print(spam.minimums())
print(spam.maximums())
print(spam.modes())
print(spam.means())
print(spam.standard_deviations())
"""

"""
spam.train_test_split(0.3, True)
print(spam.training_x)
print(spam.training_y)
print(spam.testing_x)
print(spam.testing_y)
spam.k_fold(10)

fold = 0
for training_index, validation_index in spam.k_fold_splits:
    print("Fold: {}".format(fold))
    training_x = spam.training_x[training_index]
    training_y = spam.training_y[training_index]
    validation_x = spam.training_x[validation_index]
    validation_y = spam.training_y[validation_index]

    print(training_x)
    print(training_y)
    print(validation_x)
    print(validation_y)

    fold += 1
"""

"""
# print(spam.pca_reduce(2))
# print(spam.lda_reduce(1))
"""

"""
spam.pca_graph()
spam.lda_graph()
spam.boxplot()
"""
