import tensorflow as tf
from cifar import cifar10


class cnnCIFAR10():
    def __init__(self, data, learning_rate, epochs, batch_size, weight_stddev, bias_init, keep_prob):
        self.data = data

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_stddev = weight_stddev
        self.bias_init = bias_init
        self.keep_prob = keep_prob

        self.x = None
        self.h_pool2_flat = None
        self.keep_probability = None
        self.y = None
        self.y_ = None
        self.train_step = None
        self.accuracy = None

        self.build_graph()
        self.build_eval()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 3072])
        x_image = tf.reshape(self.x, [-1, 32, 32, 3])

        # convolution 1
        W_conv1 = self.weight_variable([5, 5, 3, 32])  # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)  # ?x32x32x32
        h_pool1 = self.max_pool_2x2(h_conv1)  # ?x16x16x32

        # convolution 2
        W_conv2 = self.weight_variable([5, 5, 32, 64])  # second conv-layer has 64 kernels, size=5
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)  # ?x16x16x64
        h_pool2 = self.max_pool_2x2(h_conv2)  # ?x8x8x64

        # densely/fully connected layer
        W_fc1 = self.weight_variable([8 * 8 * 64, 1024])  # 4096x1024
        b_fc1 = self.bias_variable([1024])

        self.h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])  # ?x4096
        h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, W_fc1) + b_fc1)  # ?x1024

        # dropout regularization
        self.keep_probability = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_probability)  # ?x1024

        # linear classifier
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])  # ?x10
        self.y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2  # ?x10

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=self.weight_stddev)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(self.bias_init, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def build_eval(self):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        # Open the session and initialize variables as specified (random, 0, etc.)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # Iterate through random batches from the training data
        for i in range(self.epochs):
            batch_x, batch_y = self.data.next_batch(self.batch_size)

            # train the model
            sess.run(self.train_step,
                     feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_probability: self.keep_prob})

            # intermittently evaluate training accuracy
            if i % 10 == 0:
                training_accuracy = sess.run(self.accuracy,
                                             feed_dict={self.x: batch_x, self.y_: batch_y, self.keep_probability: 1.0})

                print("At step {}, training accuracy is {:.0f}%".format(i, training_accuracy * 100))

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, 'LeNet5_model', global_step=self.epochs)

    def validation_eval(self, model_name):
        # Load saved model
        sess = tf.Session()
        saver = tf.train.import_meta_graph(model_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Evaluate the model using the current validation data
        validation_acc = sess.run(self.accuracy, feed_dict={
            self.x: self.data.validation_x, self.y_: self.data.validation_y, self.keep_probability: 1.0})

        return validation_acc

    def test_eval(self, model_name):
        # Load saved model
        sess = tf.Session()
        saver = tf.train.import_meta_graph(model_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Evaluate the model using the testing data
        test_acc = sess.run(self.accuracy, feed_dict={
            self.x: self.data.testing_x, self.y_: self.data.testing_y, self.keep_probability: 1.0})

        return test_acc

    def encode_features(self, model_name):
        # Load saved model
        sess = tf.Session()
        saver = tf.train.import_meta_graph(model_name)
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        # Evaluate the model using all the data; return the features before the fully connected layers
        features = sess.run(self.h_pool2_flat, feed_dict={
            self.x: self.data.all_x, self.y_: self.data.all_y, self.keep_probability: 1.0})

        return features

if __name__ == '__main__':
    # Load data
    cifar = cifar10("cifar-10-batches-py")

    """
    # 5-fold Cross validation for hyperparameter testing
    # (run this first, stop program, then run the training stuff later)
    validation_accuracy = 0
    for validation_batch in range(0, 5):
        cifar.cross_validation(validation_batch)

        cnn = cnnCIFAR10(learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                         weight_stddev=weight_stddev, bias_init=bias_init,
                         dropout_prob=dropout_prob)
        cnn.train()
        validation_accuracy += cnn.validation_eval()

    validation_accuracy /= 5.0
    """

    # Load full training data
    cifar.full_training()

    # Set parameters learned from cross validation
    learning_rate = 0.0001
    epochs = 200
    batch_size = 50
    weight_stddev = 0.01  # for normal weights
    bias_init = 0.1
    keep_prob = 0.8  # for dropout

    # setup model
    cnn = cnnCIFAR10(data=cifar, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size,
                     weight_stddev=weight_stddev, bias_init=bias_init, keep_prob=keep_prob)

    # Train using the full training data
    cnn.train()

    # Test data accuracy
    cifar.testing()
    testing_accuracy = cnn.test_eval("LeNet5_model-{}.meta".format(epochs))
    print("Testing accuracy is {:.0f}%".format(testing_accuracy * 100))

    """
    # Run convolution layers to encode the images as features
    # (for use in other classification methods, slow af with all 60000 samples)
    features = cnn.encode_features("LeNet5_model-{}.meta".format(epochs))
    np.save("allfeatures.npy", features)
    """





