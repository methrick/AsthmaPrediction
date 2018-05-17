import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops


class SVMModel:
    # Create graph
    sess = ''
    features_size = 2 * 480
    x_vals: np.ndarray
    y_vals: np.ndarray

    def __init__(self, features, labels):
        ops.reset_default_graph()
        self.y_vals = np.array(labels)
        self.x_vals = np.array(
            [self.normalize_data(np.concatenate((x['ASE'].reshape(-1, 1), x['Ti'].reshape(-1, 1)), axis=0))
             for i, x in enumerate(features)])
        self.sess = tf.Session()

    def train_the_system(self):
        # Declare batch size
        batch_size = int(self.y_vals.size / 2)

        # Initialize placeholders
        x_data = tf.placeholder(shape=[None, self.features_size], dtype=tf.float32)
        y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        prediction_grid = tf.placeholder(shape=[None, self.features_size], dtype=tf.float32)

        # Create variables for svm
        b = tf.Variable(tf.random_normal(shape=[1, batch_size]))

        # Apply kernel
        # Linear Kernel
        # my_kernel = tf.matmul(x_data, tf.transpose(x_data))

        # Gaussian (RBF) kernel
        gamma = tf.constant(-50.0)
        dist = tf.reduce_sum(tf.square(x_data), 1)
        dist = tf.reshape(dist, [-1, 1])
        sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))),
                          tf.transpose(dist))
        my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

        # Compute SVM Model
        first_term = tf.reduce_sum(b)
        b_vec_cross = tf.matmul(tf.transpose(b), b)
        y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
        second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
        loss = tf.negative(tf.subtract(first_term, second_term))

        # Create Prediction Kernel
        # Linear prediction kernel
        # my_kernel = tf.matmul(x_data, tf.transpose(prediction_grid))

        # Gaussian (RBF) prediction kernel
        rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])  # Convert it to column vector
        rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])  # Convert it to column vector
        pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                              tf.transpose(rB))  #
        pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

        prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target), b), pred_kernel)
        prediction = tf.sign(prediction_output - tf.reduce_mean(prediction_output))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(y_target)), tf.float32))

        # Declare optimizer
        my_opt = tf.train.GradientDescentOptimizer(0.002)
        train_step = my_opt.minimize(loss)

        # Initialize variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Training loop
        loss_vec = []
        batch_accuracy = []
        epoch = 1000
        for i in range(epoch):
            rand_index = np.random.choice(len(self.x_vals), size=batch_size)
            rand_x = self.x_vals[rand_index]
            rand_y = np.transpose([self.y_vals[rand_index]])
            self.sess.run(train_step, feed_dict=***REMOVED***x_data: rand_x, y_target: rand_y***REMOVED***)

            temp_loss = self.sess.run(loss, feed_dict=***REMOVED***x_data: rand_x, y_target: rand_y***REMOVED***)
            loss_vec.append(temp_loss)

            acc_temp = self.sess.run(accuracy, feed_dict=***REMOVED***x_data: rand_x,
                                                          y_target: rand_y,
                                                          prediction_grid: rand_x***REMOVED***)
            batch_accuracy.append(acc_temp)

            if (i + 1) % 250 == 0:
                print('Step #' + str(i + 1))
                print('Loss = ' + str(temp_loss))

        # Plot batch accuracy
        plt.plot(batch_accuracy, 'k-', label='Accuracy')
        plt.title('Batch Accuracy')
        plt.xlabel('Generation')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        # Plot loss over time
        plt.plot(loss_vec, 'k-')
        plt.title('Loss per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.show()

    def normalize_data(self, data):
        min = data.min()
        max = data.max()
        normalized_data = np.array([2 * (x - min) / (max - min) - 1 for i, x in enumerate(data)]).reshape(-1,)
        return normalized_data
