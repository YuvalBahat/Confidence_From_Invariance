import tensorflow as tf
import numpy as np

STDDEV = 0.005  # 0.01#0.05

def FC_layer(x, W, b, keep_prob=1, name=None, bn_learning=None):
    x = tf.matmul(x, W)
    x = tf.nn.relu(tf.nn.bias_add(x, b), name=name)
    if bn_learning is not None:
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=bn_learning, scope='bn')
    return tf.nn.dropout(x, keep_prob)

class Detector_NN:
    def __init__(self, features_vect, layers_widths, keep_prob, bn_learning):
        with tf.name_scope('Detector') as global_scope:
            weights,biases = [],[]
            layers = [features_vect]
            layers_widths_ = [int(features_vect.shape[-1])] + layers_widths
            with tf.variable_scope(global_scope+'layer_0') as scope:
                weights.append(tf.Variable(tf.truncated_normal([int(layers_widths_[0]),int(layers_widths_[1])],stddev=STDDEV), name='weights'))
                biases.append(tf.Variable(tf.zeros([layers_widths_[1]]), name='biases'))
                layers.append(FC_layer(tf.reshape(layers[0], [-1, np.prod(layers[-1].get_shape().as_list()[1:])]),
                    weights[0], biases[0], keep_prob=keep_prob, name=scope.name,bn_learning=bn_learning))
            for layer_num in range(1, len(layers_widths_)):
                with tf.variable_scope(global_scope+'layer_%d' % (layer_num)) as scope:
                    if layer_num < len(layers_widths):
                        weights.append(tf.Variable(tf.truncated_normal([int(layers_widths_[layer_num]),
                            int(layers_widths_[layer_num + 1])],stddev=STDDEV), name='weights'))
                        biases.append(tf.Variable(tf.zeros([layers_widths_[layer_num + 1]]), name='biases'))
                        layers.append(FC_layer(tf.reshape(layers[layer_num], [-1, np.prod(layers[-1].get_shape().as_list()[1:])]),
                             weights[-1], biases[-1], keep_prob=keep_prob, name=scope.name,bn_learning=bn_learning))
            with tf.variable_scope(global_scope+'sigmoid') as scope:
                weights.append(tf.Variable(tf.truncated_normal(
                    [int(layers_widths_[-1]), 1], stddev=STDDEV), name='weights'))
                biases.append(tf.Variable(tf.zeros([1]), name='biases'))
                self.logit = (tf.reshape(tf.nn.bias_add(tf.matmul(layers[-1], weights[-1]), biases[-1]), [-1]))
                layers.append(tf.sigmoid(self.logit, name=scope.name))
            self.output = tf.reshape(layers[-1], [-1])
            self.num_weights = sum([np.prod(weight.get_shape().as_list()) for weight in weights]) + sum([np.prod(weight.get_shape().as_list()) for weight in biases])
