
#### Written by Jeongrae Kim in KIST

import tensorflow as tf
import numpy as np
from numpy import genfromtxt

tf.set_random_seed(123)
np.random.seed(123)

class Autoencoder:
    def __init__(self, input_dim, hidden_dim, output_dim, epoch=1000, learning_rate=0.0001, batch_size=32):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
        y = tf.placeholder(dtype=tf.float32, shape=[None, output_dim])

        with tf.name_scope('encode'):
            weights_e = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')
            biases_e = tf.Variable(tf.zeros([hidden_dim]), name='biases')
            encoded = tf.nn.tanh(tf.add(tf.matmul(x, weights_e), biases_e))

        with tf.name_scope('decode'):
            weights_d = tf.Variable(tf.random_normal([hidden_dim, output_dim], dtype=tf.float32), name='weights')
            biases_d = tf.Variable(tf.zeros([output_dim]), name='biases')
            decoded = tf.nn.tanh(tf.add(tf.matmul(encoded, weights_d), biases_d))

        self.x = x
        self.y = y
        self.encoded = encoded
        self.decoded = decoded
        self.w_encoder = weights_e

        self.loss = tf.losses.mean_squared_error(self.x, self.decoded)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.saver = tf.train.Saver()

    def get_batch(self, X, size):
        a = np.random.choice(len(X), size, replace=False)
        return X[a]

    def train_XY(self, data_X, data_Y):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            epoch_cost_list = []
            for i in range(self.epoch):
                batch_cost_ = 0
                for j in range(np.shape(data_X)[0] // self.batch_size):
                    batch_data_X = self.get_batch(data_X, self.batch_size)
                    batch_data_Y = self.get_batch(data_Y, self.batch_size)
                    l, _ , w_e= sess.run([self.loss, self.train_op, self.w_encoder], feed_dict={self.x: batch_data_X, self.y: batch_data_Y})
                    batch_cost_ += l
                epoch_cost_list.append(batch_cost_/self.batch_size)

                if i % 10 == 0:
                    print('epoch {0}: loss = {1}'.format(i, batch_cost_/self.batch_size))

                if i == (self.epoch-1):
                    weight_hidden = w_e
                    file_name_1 = './2_ChemicalElementVector.csv'
                    np.savetxt(file_name_1, weight_hidden, delimiter=",")

dir_X_file = './1_CompositionMatrix_TFIDF.csv'
data_X_ = genfromtxt(dir_X_file , delimiter=',')
data_X = data_X_[1:,]
input_dim = len(data_X[0])
dir_Y_file = './1_CompositionMatrix_TFIDF.csv'
data_Y_ = genfromtxt(dir_Y_file  , delimiter=',')
data_Y = data_Y_[1:,]
output_dim = len(data_Y[0])

hidden_dim = 30
ae = Autoencoder(input_dim, hidden_dim, output_dim )
ae.train_XY(data_X, data_Y)
