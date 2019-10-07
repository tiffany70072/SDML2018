"""The model is modified from the tensorflow tutorial: https://www.tensorflow.org/get_started/mnist/pros
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder(tf.float32, shape = [None, 8*5+4])
    self.y_input = tf.placeholder(tf.int64, shape = [None])

    # First fully connected layer.
    W_fc1 = self._weight_variable([8*5+4, 256])
    b_fc1 = self._bias_variable([256])
    
    h_fc1 = tf.nn.relu(tf.matmul(self.x_input, W_fc1) + b_fc1)

    # Output layer.
    W_fc2 = self._weight_variable([256, 256])
    b_fc2 = self._bias_variable([256])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = self._weight_variable([256, 2])
    b_fc3 = self._bias_variable([2])
    self.pre_softmax = tf.matmul(h_fc2, W_fc3) + b_fc3

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y_input, logits=self.pre_softmax)
    self.xent = tf.reduce_sum(y_xent)
    self.y_pred = tf.argmax(self.pre_softmax, 1)
    correct_prediction = tf.equal(self.y_pred, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
