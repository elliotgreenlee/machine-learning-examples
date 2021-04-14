"""
/Users/elliot/Documents/Class/692/projects/project3/venv_denoising_autoencoder/bin/conda install -p /Users/elliot/Documents/Class/692/projects/project3/venv_denoising_autoencoder sklearn -y
"""
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import numpy as np

"""
# CNN
# Load data
features = np.load("allfeatures.npy")
training_features = features[0:50000]
testing_features = features[50000:60000]
trY = np.load("trY.npy")
teY = np.load("teY.npy")

# Train SVM
print("fitting")
clf = svm.SVC()
clf.fit(training_features, trY)
print("fitted")

# Predict
print("predicting")
prY = clf.predict(testing_features)
print("predicted")

# Show results
print("CNN Accuracy: {}".format(accuracy_score(teY, prY)))
print(prY)
"""

# Autoencoder
# Load data
features = np.load("dae-all.npy")
training_features = features[0:50000]
testing_features = features[50000:60000]
trY = np.load("trY.npy")
teY = np.load("teY.npy")

# Train SVM
print("fitting")
clf = svm.SVC()
clf.fit(training_features, trY)
print("fitted")

# Predict
print("predicting")
prY = clf.predict(testing_features)
print("predicted")

# Show results
print("DAE Accuracy: {}".format(accuracy_score(teY, prY)))
print(prY)

joblib.dump(clf, 'dae_features.pkl')
# clf = joblib.load('lenet_part_features.pkl')



