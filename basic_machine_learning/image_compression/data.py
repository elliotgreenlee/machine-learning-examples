from scipy.misc import imread, imsave
import random
import math
import sys
import copy
import time

RED = 0
GREEN = 1
BLUE = 2


# clustering
class Unsupervised:
    def __init__(self, clusters, original_image):
        self.clusters = clusters
        self.original_image = original_image
        self.image = original_image.copy()
        self.image_width = len(original_image)
        self.image_height = len(original_image[0])

        self.classification = [[0 for x in range(self.image_width)] for y in range(self.image_height)]  # every pixel location will have a classification from 0 to clusters-1

        # Generate original cluster centers
        self.cluster_centers = [[0 for x in range(3)] for y in range(self.clusters)]  # clusters x 3 vector with each randomly generated cluster center

        for center in self.cluster_centers:
            center[RED] = random.randint(0, 255)
            center[GREEN] = random.randint(0, 255)
            center[BLUE] = random.randint(0, 255)

    @staticmethod
    # TODO: is this math ok, especially for the kohonen?
    def euclidean(a, b):
        if len(a) != len(b):
            print("Euclidean math needs help")
            exit(1)

        sum = 0
        for i in range(len(a)):
            diff = a[i] - b[i]
            sum += math.pow(diff, 2)

        return math.sqrt(sum)

    # Assigns every image pixel to the closest center
    def assign_samples(self, distance):

        for pixel_i in range(self.image_height):
            for pixel_j in range(self.image_width):
                min_distance = sys.maxint
                min_cluster = -1
                for cluster, center in enumerate(self.cluster_centers):
                    new_distance = distance(center, self.image[pixel_i][pixel_j])
                    if new_distance < min_distance:
                        min_distance = new_distance
                        min_cluster = cluster

                self.classification[pixel_i][pixel_j] = min_cluster

    def different(self, old_classification, new_classification):
        for pixel_i in range(self.image_height):
            for pixel_j in range(self.image_width):
                if old_classification[pixel_i][pixel_j] != new_classification[pixel_i][pixel_j]:
                    print "Difference found: ", pixel_i, pixel_j, old_classification[pixel_i][pixel_j], new_classification[pixel_i][pixel_j]
                    return True

        return False

    # Compute the average difference between pixels in the original image and its compression
    def compression_error(self):
        diff = [0] * 3
        sum = 0
        for pixel_i in range(self.image_height):
            for pixel_j in range(self.image_width):
                diff[RED] = self.original_image[pixel_i][pixel_j][RED] - self.image[pixel_i][pixel_j][RED]
                diff[GREEN] = self.original_image[pixel_i][pixel_j][GREEN] - self.image[pixel_i][pixel_j][GREEN]
                diff[BLUE] = self.original_image[pixel_i][pixel_j][BLUE] - self.image[pixel_i][pixel_j][BLUE]

                diff_total = abs(diff[RED]) + abs(diff[GREEN]) + abs(diff[BLUE])

                sum += pow(diff_total, 2)

        average = sum / (self.image_width * self.image_height)

        return math.sqrt(average)

    def reduce_image(self, picture_name):
        print "Reducing Image"
        for pixel_i in range(self.image_height):
            for pixel_j in range(self.image_width):
                self.image[pixel_i][pixel_j] = self.cluster_centers[self.classification[pixel_i][pixel_j]]
        imsave(picture_name, self.image)

    def kmeans_cluster(self):
        # store old classification
        old_classification = copy.deepcopy(self.classification)
        # assign samples
        self.assign_samples(self.euclidean)
        print "Initial assignment done"

        timing_sum = 0
        # while there are differences
        iteration = 1
        while self.different(old_classification, self.classification):
            start_time = time.time()
            print("Iteration: ", iteration)
            iteration += 1

            # store old classification
            old_classification = copy.deepcopy(self.classification)
            # calculate new cluster centers
            self.kmeans_centers()
            # assign samples
            self.assign_samples(self.euclidean)

            timing_sum += time.time() - start_time
            print time.time() - start_time

        timing_average = timing_sum / (iteration * 1.0)

        with open('kmeans_performance.txt', 'a') as f:
            f.write('{} {} {}\n'.format(self.clusters, iteration, timing_average))

    def kmeans_centers(self):
        # vector of RGB clusters initialized to 0
        for center in self.cluster_centers:
            center[RED] = 0
            center[GREEN] = 0
            center[BLUE] = 0

        # iterate over pixels
        mean_counts = [0] * self.clusters
        for pixel_i in range(self.image_height):
            for pixel_j in range(self.image_width):
                cluster = self.classification[pixel_i][pixel_j]
                # add pixel values to that index in vector of RGB clusters
                mean_counts[cluster] += 1
                self.cluster_centers[cluster][RED] += self.image[pixel_i][pixel_j][RED]
                self.cluster_centers[cluster][GREEN] += self.image[pixel_i][pixel_j][GREEN]
                self.cluster_centers[cluster][BLUE] += self.image[pixel_i][pixel_j][BLUE]

        # divide by number of samples in cluster
        for cluster, center in enumerate(self.cluster_centers):
            if mean_counts[cluster] == 0:
                mean_counts[cluster] = 1
            center[RED] /= mean_counts[cluster]
            center[GREEN] /= mean_counts[cluster]
            center[BLUE] /= mean_counts[cluster]

    def winner_cluster(self):
        # store old classification
        old_classification = copy.deepcopy(self.classification)
        # assign samples
        start_time = time.time()
        self.assign_samples(self.euclidean)
        print "Initial assignment done"

        timing_sum = 0
        # while there are differences
        iteration = 1
        while self.different(old_classification, self.classification):
            start_time = time.time()
            print("Iteration: ", iteration)
            iteration += 1

            # store old classification
            old_classification = copy.deepcopy(self.classification)
            # calculate new cluster centers
            self.winner_centers()
            # assign samples
            self.assign_samples(self.euclidean)

            timing_sum += time.time() - start_time
            time.time() - start_time

        timing_average = timing_sum / (iteration * 1.0)

        with open('winner_performance.txt', 'a') as f:
            f.write('{} {} {}\n'.format(self.clusters, iteration, timing_average))

    def winner_centers(self, learning_rate=0.01):
        # iterate over pixels
        for pixel_i in range(self.image_height):
            for pixel_j in range(self.image_width):
                cluster = self.classification[pixel_i][pixel_j]
                # add pixel values to that index in vector of RGB clusters
                self.cluster_centers[cluster][RED] += learning_rate * (self.image[pixel_i][pixel_j][RED] - self.cluster_centers[cluster][RED])
                self.cluster_centers[cluster][GREEN] += learning_rate * (self.image[pixel_i][pixel_j][GREEN] - self.cluster_centers[cluster][GREEN])
                self.cluster_centers[cluster][BLUE] += learning_rate * (self.image[pixel_i][pixel_j][BLUE] - self.cluster_centers[cluster][BLUE])

    def kohonen_cluster(self):
        # store old classification
        old_classification = copy.deepcopy(self.classification)
        # assign samples
        self.assign_samples(self.euclidean)
        print "Initial assignment done"

        timing_sum = 0
        # while there are differences
        iteration = 1
        while self.different(old_classification, self.classification):
            start_time = time.time()
            print("Iteration: ", iteration)
            iteration += 1

            # store old classification
            old_classification = copy.deepcopy(self.classification)
            # calculate new cluster centers
            self.kohonen_centers()
            # assign samples
            self.assign_samples(self.euclidean)

            timing_sum += time.time() - start_time
            time.time() - start_time

        timing_average = timing_sum / (iteration * 1.0)

        with open('kohonen_performance.txt', 'a') as f:
            f.write('{} {} {}\n'.format(self.clusters, iteration, timing_average))

    def kohonen_centers(self, learning_rate=0.01):
        # iterate over pixels
        for pixel_i in range(self.image_height):
            for pixel_j in range(self.image_width):
                winning_cluster = self.classification[pixel_i][pixel_j]
                for index, cluster in enumerate(self.cluster_centers):
                    cluster[RED] += learning_rate * self.closeness(winning_cluster, index) * (self.image[pixel_i][pixel_j][RED] - cluster[RED])
                    cluster[GREEN] += learning_rate * self.closeness(winning_cluster, index) * (self.image[pixel_i][pixel_j][GREEN] - cluster[GREEN])
                    cluster[BLUE] += learning_rate * self.closeness(winning_cluster, index) * (self.image[pixel_i][pixel_j][BLUE] - cluster[BLUE])

    def closeness(self, winning_cluster, other_cluster, variance=1.0):
        return math.exp((-1.0 * pow(self.topological_distance(winning_cluster, other_cluster), 2.0)) / (2.0 * variance))

    def topological_distance(self, winning_cluster, other_cluster):
        return winning_cluster - other_cluster

# Read in the image
original_image = imread('flowers.ppm')

# K-means
'''
for i in range(0, 9):
    clusters = pow(2, i)
    print "K-means, {} clusters".format(clusters)
    # kmeans = Unsupervised(clusters, copy.deepcopy(original_image))
    kmeans.kmeans_cluster()
    kmeans.reduce_image('images/k-means{}.ppm'.format(clusters))
'''

# Winner take all
'''
for i in range(0, 9):
    clusters = pow(2, i)
    print "Winner, {} clusters".format(clusters)
    winner = Unsupervised(clusters, copy.deepcopy(original_image))
    winner.winner_cluster()
    winner.reduce_image('images/winner{}.ppm'.format(clusters))
'''

# Kohonen
'''
for i in range(0, 9):
    clusters = pow(2, i)
    print "Kohonen, {} clusters".format(clusters)
    kohonen = Unsupervised(clusters, copy.deepcopy(original_image))
    kohonen.kohonen_cluster()
    kohonen.reduce_image('images/kohonen{}.ppm'.format(clusters))
'''

# Mean shift
import numpy as np
from sklearn.cluster import MeanShift

window_sizes = [11, 12, 13, 14]
for window_size in window_sizes:
    print "Mean Shift, window size {}".format(window_size)
    ms = MeanShift(bandwidth=window_size, bin_seeding=True)

    copy_image = copy.deepcopy(original_image)

    X = np.zeros((480*480, 3))
    # iterate over pixels
    for pixel_i in range(480):
        for pixel_j in range(480):
            X[pixel_i * 480 + pixel_j][RED] = copy_image[pixel_i][pixel_j][RED]
            X[pixel_i * 480 + pixel_j][GREEN] = copy_image[pixel_i][pixel_j][GREEN]
            X[pixel_i * 480 + pixel_j][BLUE] = copy_image[pixel_i][pixel_j][BLUE]

    print X

    start_time = time.time()
    ms.fit(X)

    labels = ms.labels_
    print labels
    cluster_centers = ms.cluster_centers_
    with open('means_performance.txt', 'a') as f:
        f.write('{} {} {}\n'.format(window_size, time.time() - start_time, len(cluster_centers)))
    print("number of estimated clusters : {}".format(len(cluster_centers)))
    print cluster_centers

    print "Reducing Image"
    for pixel_i in range(480):
        for pixel_j in range(480):
            copy_image[pixel_i][pixel_j][RED] = cluster_centers[labels[pixel_i * 480 + pixel_j]][RED]
            copy_image[pixel_i][pixel_j][GREEN] = cluster_centers[labels[pixel_i * 480 + pixel_j]][GREEN]
            copy_image[pixel_i][pixel_j][BLUE] = cluster_centers[labels[pixel_i * 480 + pixel_j]][BLUE]
    imsave('mean_shift{}.ppm'.format(window_size), copy_image)




