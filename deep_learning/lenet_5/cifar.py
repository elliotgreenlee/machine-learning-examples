import numpy as np


def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


# There are 50000 training images and 10000 test images.
class cifar10():
    def __init__(self, directory_name):
        # Load in the data
        self.directory_name = directory_name
        self.label_names = unpickle("{}/batches.meta".format(self.directory_name))

        self.train_batches = []
        self.train_batches.append(unpickle("{}/data_batch_1".format(self.directory_name)))
        self.train_batches.append(unpickle("{}/data_batch_2".format(self.directory_name)))
        self.train_batches.append(unpickle("{}/data_batch_3".format(self.directory_name)))
        self.train_batches.append(unpickle("{}/data_batch_4".format(self.directory_name)))
        self.train_batches.append(unpickle("{}/data_batch_5".format(self.directory_name)))

        self.test_batches = []
        self.test_batches.append(unpickle("{}/test_batch".format(self.directory_name)))

        # Initialize images and labels
        self.training_x = None
        self.training_y = None
        self.validation_x = None
        self.validation_y = None
        self.testing_x = None
        self.testing_y = None
        self.all_x = None
        self.all_y = None

    # Prepare data for training
    def full_training(self):
        self.training_x = []
        self.training_y = []

        for batch in self.train_batches:
            for pair in zip(batch['data'], batch['labels']):
                image = pair[0].astype(np.float32)
                label = np.zeros(10)
                label[pair[1]] = 1
                self.training_x.append(image)
                self.training_y.append(label)

        self.training_x = np.asarray(self.training_x)
        self.training_y = np.asarray(self.training_y)

    # Prepare data for cross validation
    def cross_validation(self, validation_batch):
        self.training_x = []
        self.training_y = []
        self.validation_x = []
        self.validation_y = []

        for i, batch in enumerate(self.train_batches):
            if i is not validation_batch:
                for pair in zip(batch['data'], batch['labels']):
                    image = pair[0].astype(np.float32)
                    label = np.zeros(10)
                    label[pair[1]] = 1
                    self.training_x.append(image)
                    self.training_y.append(label)
            else:
                for pair in zip(batch['data'], batch['labels']):
                    image = pair[0].astype(np.float32)
                    label = np.zeros(10)
                    label[pair[1]] = 1
                    self.validation_x.append(image)
                    self.validation_y.append(label)

        self.training_x = np.asarray(self.training_x)
        self.training_y = np.asarray(self.training_y)
        self.validation_x = np.asarray(self.validation_x)
        self.validation_y = np.asarray(self.validation_y)

    # create next batch of given size
    def next_batch(self, batch_size):
        all_indices = np.arange(0, len(self.training_x))  # get all possible indices
        np.random.shuffle(all_indices)  # shuffle all indices randomly
        batch_indices = all_indices[0:batch_size]  # use the first batch_size indices

        batch_x = []
        batch_y = []

        for i in batch_indices:
            batch_x.append(self.training_x[i])
            batch_y.append(self.training_y[i])

        return np.asarray(batch_x), np.asarray(batch_y)

    # Prepare data for testing
    def testing(self):
        self.testing_x = []
        self.testing_y = []

        for batch in self.test_batches:
            for pair in zip(batch['data'], batch['labels']):
                image = pair[0].astype(np.float32)
                label = np.zeros(10)
                label[pair[1]] = 1
                self.testing_x.append(image)
                self.testing_y.append(label)

        self.testing_x = np.asarray(self.testing_x)
        self.testing_y = np.asarray(self.testing_y)

    # Prepare data for feature encoding
    def feature_encoding(self):
        self.all_x = []
        self.all_y = []

        for i, batch in enumerate(self.train_batches):
            for pair in zip(batch['data'], batch['labels']):
                image = pair[0].astype(np.float32)
                label = np.zeros(10)
                label[pair[1]] = 1
                self.all_x.append(image)
                self.all_y.append(label)

        for batch in self.test_batches:
            for pair in zip(batch['data'], batch['labels']):
                image = pair[0].astype(np.float32)
                label = np.zeros(10)
                label[pair[1]] = 1
                self.all_x.append(image)
                self.all_y.append(label)

        self.all_x = np.asarray(self.all_x)
        self.all_y = np.asarray(self.all_y)
