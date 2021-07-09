
#### Written by Jeongrae Kim in KIST

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from PIL import Image

tf.set_random_seed(123)
np.random.seed(123)
Material_vector_number = 60
num_MV_hidden_node = 512
IMG_H = 128
IMG_W = 128
num_DP_hidden_node = 512
num_Mv_hat = 9
num_hidden_node_MsDNN = 10
num_DP_hat = num_hidden_node_MsDNN - num_Mv_hat

class CNNs(object):
    def __init__(self, sess):
        self.sess = sess
        self.class_num = 1
        self.set_dropout_Mv = 0.7
        self.set_dropout_DP = 0.7
        self.learning_rate = 0.0001
        self.height = IMG_H
        self.width = IMG_W
        self.channel = 3
        self.num_DP_hat = num_DP_hat
        self.material_vector_num = Material_vector_number
        self.num_MV_hidden_node = num_MV_hidden_node
        self.num_DP_hidden_node = num_DP_hidden_node
        self.num_Mv_hat = num_Mv_hat
        self.num_hidden_node_MsDNN = num_hidden_node_MsDNN
        self._build_net()
        print('Initialized networks!')

    def _build_net(self):
        self.X_Ef = tf.placeholder(tf.float32, [None, self.width, self.height, self.channel])
        #
        self.CNN_material_vector_thph = tf.placeholder(tf.float32, [None, self.material_vector_num, 1, 1], name="Inputs")
        self.actual_y = tf.placeholder(tf.float32, shape=[None, 1], name='P')
        self.drop_prob_MV = tf.placeholder(tf.float32, name='keep_prob')
        self.drop_prob_DP = tf.placeholder(tf.float32, name='keep_prob')
        self.pred = self.multi_input_CNN(self.X_Ef, self.CNN_material_vector_thph)
        self.MAE = tf.losses.absolute_difference(self.actual_y, self.pred)
        self.MSE = tf.losses.mean_squared_error(self.actual_y, self.pred)
        self.total_loss = tf.losses.absolute_difference(self.actual_y, self.pred)
        self.train_op_pred = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss/2)

    def conv2d(x, output_dim, k_h=5, k_w=5, d_h=1, d_w=1, stddev=0.02, padding='SAME', name='conv2d_1'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_ = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv_, biases)
            return conv

    def conv2d_CNN(x, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, padding='SAME', name='conv2d_2'):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, x.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv_ = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv_, biases)
            return conv

    def relu(x, name='relu'):
        output = tf.nn.relu(x, name=name)
        return output

    def selu(x, name='selu'):
        output = tf.nn.selu(x, name=name)
        return output

    def elu(x, name='elu'):
        output = tf.nn.elu(x, name=name)
        return output

    def relu6(x, name='relu6'):
        output = tf.nn.relu6(x, name=name)
        return output

    def tanh(x, name='tanh'):
        output = tf.nn.tanh(x, name=name)
        return output

    def max_pool_2x1_CNN(x, name='max_pool'):
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

    def max_pool_2x2(x, name='max_pool'):
        with tf.name_scope(name):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def avg_pool_2x2(x, name='max_pool'):
        with tf.name_scope(name):
            return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def fc(x, output_size, bias_start=0.0, with_w=False, name='fc'):
        shape = x.get_shape().as_list()
        with tf.variable_scope(name):
            matrix = tf.get_variable(name="matrix", shape=[shape[1], output_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="bias", shape=[output_size], initializer=tf.constant_initializer(bias_start))
            if matrix.shape[1]!=0:
                if with_w and bias.shape!=0:
                    return tf.matmul(x, matrix) + bias, matrix, bias
                if bias.shape!=0:
                    return tf.matmul(x, matrix) + bias

    def multi_input_CNN(self, DP, material_vector_AB):
        #tf.set_random_seed(123)
        #tf.contrib.layers.xavier_initializer(seed=123)
        net_Ef_1 = CNNs.conv2d(DP, 32, name='conv2d_1')
        net_Ef_1_p = CNNs.max_pool_2x2(net_Ef_1)
        net_Ef_2 = CNNs.conv2d(net_Ef_1_p, 32, name='conv2d_2')
        net_Ef_2_p = CNNs.max_pool_2x2(net_Ef_2)
        net_Ef_3 = CNNs.conv2d(net_Ef_2_p, 32, name='conv2d_3')
        net_Ef_3_p = CNNs.max_pool_2x2(net_Ef_3)
        DP_fs = tf.layers.flatten(net_Ef_3_p)
        dp_fc14_3_relu = CNNs.relu(DP_fs, name='fc14_relu')
        dp_fc15_3 = CNNs.fc(dp_fc14_3_relu, self.num_DP_hidden_node, name='fc15')
        dp_fc15_3_relu = CNNs.relu(dp_fc15_3, name='fc15_relu')
        dp_fc15_3_drop = tf.nn.dropout(dp_fc15_3_relu, self.drop_prob_DP)
        DP_out = CNNs.fc(dp_fc15_3_drop, self.num_DP_hat, name='fc15_dp')
        conv1_1 = CNNs.conv2d_CNN(material_vector_AB, 64, name='CNN_conv1', k_h=3, k_w=1, d_h=1, d_w=1)
        maxpool1_1 = CNNs.max_pool_2x1_CNN(conv1_1)
        conv2_1 = CNNs.conv2d_CNN(maxpool1_1, 64, name='CNN_conv2', k_h=4, k_w=1, d_h=1, d_w=1)
        maxpool2_1 = CNNs.max_pool_2x1_CNN(conv2_1)
        conv3_1 = CNNs.conv2d_CNN(maxpool2_1, 64, name='CNN_conv3', k_h=5, k_w=1, d_h=1, d_w=1)
        maxpool3_1 = CNNs.max_pool_2x1_CNN(conv3_1)
        Mv_fs = tf.layers.flatten(maxpool3_1)
        fc14_3_relu = CNNs.relu(Mv_fs, name='CNN_fc14_relu')
        fc15_3 = CNNs.fc(fc14_3_relu, self.num_MV_hidden_node, name='CNN_fc15')
        fc15_3_relu = CNNs.relu(fc15_3, name='CNN_fc15_relu')
        fc15_3_drop = tf.nn.dropout(fc15_3_relu, self.drop_prob_MV)
        Mv_out = CNNs.fc(fc15_3_drop, self.num_Mv_hat, name='fc15_mv')
        DP_MV_vector = tf.concat([DP_out, Mv_out], 1)
        out = CNNs.fc(DP_MV_vector, self.class_num, name='fc15_')
        return out

    def train_step(self, imgs, material_vector_AB, label_batch):
        train_ops = [self.MAE, self.MSE, self.pred, self.actual_y, self.total_loss, self.train_op_pred]
        train_feed = {self.X_Ef: imgs, self.CNN_material_vector_thph: material_vector_AB,
                      self.actual_y: label_batch, self.drop_prob_MV: self.set_dropout_Mv, self.drop_prob_DP: self.set_dropout_DP}
        MAE, MSE, y_hat, y_real, loss, trn_ = self.sess.run(train_ops, feed_dict=train_feed)
        return MAE, MSE, y_hat, y_real, loss, trn_

    def test_step(self, imgs, material_vector_AB, label_batch):
        test_ops = [self.MAE, self.MSE, self.pred, self.actual_y]
        test_feed = {self.X_Ef: imgs, self.CNN_material_vector_thph: material_vector_AB,
                     self.actual_y: label_batch, self.drop_prob_MV: 1.0, self.drop_prob_DP: 1.0}
        MAE, MSE, y_hat, y_real= self.sess.run(test_ops, feed_dict=test_feed)
        return MAE, MSE, y_hat, y_real

    def data_label_train(sample, batch_size):
        feature, p_Eg, AB_material_vactor = [], [], []
        for i in range(batch_size):
            original = Image.open(sample[i][0].strip('"'))
            img_rs = original.resize((IMG_H, IMG_W))
            img = np.asarray(img_rs)
            feature.append(img / (255*2))
            p_Eg.append(float(sample[i][1].strip('"')))
            AB_material_vactor.append(sample[i][2])
        return feature, p_Eg, AB_material_vactor

    def data_label_test(sample, batch_size):
        feature, p_Eg, AB_material_vactor = [], [], []
        for i in range(batch_size):
            original = Image.open(sample[0][0].strip('"'))
            img_rs = original.resize((IMG_H, IMG_W))
            img = np.asarray(img_rs)
            feature.append(img / (255*2))
            p_Eg.append(float(sample[0][1].strip('"')))
            AB_material_vactor.append(sample[0][2])
        return feature, p_Eg, AB_material_vactor


