# DimensionalityReduction

This repository contains the code for programming assignment two for
ECE 571 at UTK as taught by Hairong Qi. 03/12/17. 

The data directory contains the testing and training data from Ripley's
Pattern Recognition and Neural Networks. 

data.py reads in and preprocesses the data using the Data class.

norm.py normalizes the data using the Norm class. 

fld.py applies Fisher's Linear Discriminant to the data to reduce the
features to one dimension.

pca.py applies Principal Component Analysis to the data to reduce the 
features by at most a given information loss error. (5 at 10%)

classify.py runs all three discriminant function cases and kNN
on given data in order to classify it.