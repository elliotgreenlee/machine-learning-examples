import numpy as np
from data import Data
from norm import Norm
from pca import PCA
from classify import Classify
from knn import KNN
from cluster import Unsupervised
import itertools

# Create Normalized data
'''
data = Data('original_features/features{}.txt')

data.load_10_data()

norm = Norm(data.features)

total = 0
for fold, num_samples in enumerate(data.feature_set):
    with open('normalized_features/normalized_features{}.txt'.format(fold), 'w') as f:
        f.write('Feature file for fold {}. Contains {} samples, each with {} features and a class\n'
                .format(fold, num_samples, len(norm.features[0])))
        for i in range(0, num_samples):
            sample = norm.features[total + i]
            for feature in sample:
                f.write('{} '.format(feature))
            f.write('{}\n'.format(data.classes[total + i]))

        total += i + 1
'''

# Reduce data
'''
data = Data('normalized_features/normalized_features{}.txt')
data.load_10_data()
max_error = 0.01
pca = PCA(data.features, max_error)

total = 0
for fold, num_samples in enumerate(data.feature_set):
    with open('pca_features/pca_features{}_{}.txt'.format(max_error, fold), 'w') as f:
        f.write('Feature file for fold {}. Contains {} samples, each with {} features and a class\n'
                .format(fold, num_samples, len(pca.features[0])))
        for i in range(0, num_samples):
            sample = pca.features[total + i]
            for feature in sample:
                f.write('{} '.format(feature))
            f.write('{}\n'.format(data.classes[total + i]))

        total += i + 1
'''

'''
data = Data('normalized_features/normalized_features{}.txt')
data.load_separate_data()
# prior_list = [[0.25, 0.25, 0.25, 0.25]]
prior_list = [[0.07226765799256506, 0.016802973977695167, 0.17618339529120197, 0.7347459727385378]]
# prior_list = [[0.1, 0.3, 0.3, 0.3], [0.3, 0.1, 0.3, 0.3], [0.3, 0.3, 0.1, 0.3], [0.3, 0.3, 0.3, 0.1]]
# prior_list = [[0.4, 0.2, 0.2, 0.2], [0.2, 0.4, 0.2, 0.2], [0.2, 0.2, 0.4, 0.2], [0.2, 0.2, 0.2, 0.4]]
# prior_list = [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]]
for priors in prior_list:
    for i in range(0, 10):
        training_features = None
        training_classes = None
        testing_features = None
        testing_classes = None
        for j in range(0, 10):
            if j == i:
                testing_features = data.features[i]
                testing_classes = data.classes[i]
            else:
                if training_features is None:
                    training_features = data.features[i]
                    training_classes = data.classes[i]
                else:
                    training_features = np.concatenate((training_features, data.features[i]), axis=0)
                    training_classes = np.concatenate((training_classes, data.classes[i]), axis=0)

        classify = Classify(training_features, testing_features, training_classes, priors)

        with open('mppresults{}_{}_{}_{}_{}_{}.txt'.format(1, 'normalized', classify.priors[0], classify.priors[1], classify.priors[2], classify.priors[3]), 'a') as f:
            for sample in classify.case1_classes:
                f.write('{}\n'.format(sample))

        with open('mppresults{}_{}_{}_{}_{}_{}.txt'.format(2, 'normalized', classify.priors[0], classify.priors[1],
                                                           classify.priors[2], classify.priors[3]), 'a') as f:
            for sample in classify.case2_classes:
                f.write('{}\n'.format(sample))

        with open('mppresults{}_{}_{}_{}_{}_{}.txt'.format(3, 'normalized', classify.priors[0], classify.priors[1],
                                                           classify.priors[2], classify.priors[3]), 'a') as f:
            for sample in classify.case3_classes:
                f.write('{}\n'.format(sample))
'''

'''
data = Data('normalized_features/normalized_features{}.txt')
data.load_separate_data()
k = 2
minkowski = 8
with open('knnresults_{}_{}_{}.txt'.format('normalized', k, minkowski), 'a') as f:
    f.write('Results file for knn. k of {} and minkowski of {}\n'.format(k, minkowski))

for i in range(0, 1):
    training_features = None
    training_classes = None
    testing_features = None
    testing_classes = None
    for j in range(0, 2):
        if j == i:
            testing_features = data.features[i]
            testing_classes = data.classes[i]
        else:
            if training_features is None:
                training_features = data.features[i]
                training_classes = data.classes[i]
            else:
                training_features = np.concatenate((training_features, data.features[i]), axis=0)
                training_classes = np.concatenate((training_classes, data.classes[i]), axis=0)
    knn = KNN(training_features, training_classes, testing_features, testing_classes, [0.25, 0.25, 0.25, 0.25], k, 44, 4, minkowski)

    with open('knnresults_{}_{}_{}.txt'.format('normalized', k, minkowski), 'a') as f:
        for sample in knn.classes:
            f.write('{}\n'.format(sample))
'''

'''
data = Data('normalized_features/normalized_features{}.txt')
data.load_10_data()

kmeans = Unsupervised(4, data.features)
kmeans.kohonen_cluster()

highest = 0
best_permute = []
permuted_list = itertools.permutations([0, 1, 2, 3])
for permutation in permuted_list:
    total = 0
    for prediction, real in zip(kmeans.classification, data.classes):
        if permutation[prediction] == real:
            total += 1
    accuracy = (1.0 * total) / len(data.classes)
    if accuracy > highest:
        highest = accuracy
        best_permute = permutation

with open('clusterresults{}.txt'.format('kohonen'), 'a') as f:
    for sample in kmeans.classification:
        f.write('{}\n'.format(best_permute[sample]))
'''
'''
data = Data('normalized_features/normalized_features{}.txt')
data.load_10_data()

with open('realresults.txt', 'w') as f:
    for sample in data.classes:
        f.write('{}\n'.format(sample))
'''

'''
#decisiontree.txt
#svm.txt
#BPNN.txt
results = []
with open('BPNN.txt') as f:
    for line in f:
        words = line.split(',')
        for word in words:
            type = int(word)
            results.append(type)

with open('BPNN_results.txt', 'w') as f:
    for sample in results:
        f.write('{}\n'.format(sample))
'''

'''
print 'Accuracies'
print 'Normalized'
print data.norm.classify.case1_accuracy
print data.norm.classify.case2_accuracy
print data.norm.classify.case3_accuracy
print data.norm.classify.knn_accuracy

print 'PCA'
print data.pca.classify.case1_accuracy
print data.pca.classify.case2_accuracy
print data.pca.classify.case3_accuracy
print data.pca.classify.knn_accuracy

print 'FLD'
print data.fld.classify.case1_accuracy
print data.fld.classify.case2_accuracy
print data.fld.classify.case3_accuracy
print data.fld.classify.knn_accuracy

print 'kNN varied'
print 'norm'
print data.norm.classify.knn_accuracies
plt.figure(1)
plt.plot(list(range(1, 21)), data.norm.classify.knn_accuracies, 'rx', markersize=4, label="Normalized")
plt.title("Normalized kNN Accuracies")
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.axis([0, 21, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})
print 'pca'
print data.pca.classify.knn_accuracies
plt.figure(2)
plt.plot(list(range(1, 21)), data.pca.classify.knn_accuracies, 'rx', markersize=4, label="PCA")
plt.title("PCA kNN Accuracies")
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.axis([0, 21, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})
print 'fld'
print data.fld.classify.knn_accuracies
plt.figure(3)
plt.plot(list(range(1, 21)), data.fld.classify.knn_accuracies, 'rx', markersize=4, label="FLD")
plt.title("FLD kNN Accuracies")
plt.xlabel("k Values")
plt.ylabel("Accuracy")
plt.axis([0, 21, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print 'prior probability varied'
print 'norm'
print data.norm.classify.case1_tprs
print data.norm.classify.case1_fprs
plt.figure(4)
plt.plot(data.norm.classify.case1_fprs, data.norm.classify.case1_tprs, 'bx', markersize=4, label="Normalized Case 1")

print data.norm.classify.case2_tprs
print data.norm.classify.case2_fprs
plt.plot(data.norm.classify.case2_fprs, data.norm.classify.case2_tprs, 'gx', markersize=4, label="Normalized Case 2")

print data.norm.classify.case3_tprs
print data.norm.classify.case3_fprs
plt.plot(data.norm.classify.case3_fprs, data.norm.classify.case3_tprs, 'rx', markersize=4, label="Normalized Case 3")

print data.norm.classify.knn_tprs
print data.norm.classify.knn_fprs
plt.plot(data.norm.classify.knn_fprs, data.norm.classify.knn_tprs, 'kx', markersize=4, label="Normalized kNN")
plt.title("Normalized ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print 'pca'
print data.pca.classify.case1_tprs
print data.pca.classify.case1_fprs
plt.figure(8)
plt.plot(data.pca.classify.case1_fprs, data.pca.classify.case1_tprs, 'bx', markersize=4, label="PCA Case 1")

print data.pca.classify.case2_tprs
print data.pca.classify.case2_fprs
plt.plot(data.pca.classify.case2_fprs, data.pca.classify.case2_tprs, 'gx', markersize=4, label="PCA Case 2")

print data.pca.classify.case3_tprs
print data.pca.classify.case3_fprs
plt.plot(data.pca.classify.case3_fprs, data.pca.classify.case3_tprs, 'rx', markersize=4, label="PCA Case 3")

print data.pca.classify.knn_tprs
print data.pca.classify.knn_fprs
plt.plot(data.pca.classify.knn_fprs, data.pca.classify.knn_tprs, 'kx', markersize=4, label="PCA kNN")
plt.title("PCA ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})

print 'fld'
print data.fld.classify.case1_tprs
print data.fld.classify.case1_fprs
plt.figure(12)
plt.plot(data.fld.classify.case1_fprs, data.fld.classify.case1_tprs, 'bx', markersize=4, label="FLD Case 1")

print data.fld.classify.case2_tprs
print data.fld.classify.case2_fprs
plt.plot(data.fld.classify.case2_fprs, data.fld.classify.case2_tprs, 'gx', markersize=4, label="FLD Case 2")

print data.fld.classify.case3_tprs
print data.fld.classify.case3_fprs
plt.plot(data.fld.classify.case3_fprs, data.fld.classify.case3_tprs, 'rx', markersize=4, label="FLD Case 3")

print data.fld.classify.knn_tprs
print data.fld.classify.knn_fprs
plt.plot(data.fld.classify.knn_fprs, data.fld.classify.knn_tprs, 'kx', markersize=4, label="FLD kNN")
plt.title("FLD ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0, 1, 0, 1])
plt.legend(loc=4, numpoints=1, prop={'size': 8})
'''

# plt.show()
