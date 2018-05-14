import tensorflow as tf
import numpy as np

STDDEV = 0.005  # 0.01#0.05

def FC_layer(x, W, b, keep_prob=1, name=None, batch_norm=None):
    x = tf.matmul(x, W)
    x = tf.nn.relu(tf.nn.bias_add(x, b), name=name)
    if batch_norm is not None:
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=batch_norm, scope='bn')
    return tf.nn.dropout(x, keep_prob)


def Conv_layer(x, W, b, keep_prob=1, name=None, batch_norm=False, conv_logits=False):
    x = tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID', name=name)
    x = tf.nn.relu(tf.nn.bias_add(x, b))
    if batch_norm is not None:
        x = tf.contrib.layers.batch_norm(x, center=True, scale=False, is_training=batch_norm, scope='bn')
    if conv_logits:
        pass
        # x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name=name + 'maxpool')
    else:
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name + 'maxpool')
    return tf.nn.dropout(x, keep_prob, name=name + 'dropout')

class Detector_NN:
    def __init__(self, features_vect, layers_widths, keep_prob, batch_norm, logits_shape,high_order_logits=None):
        high_order_logits = high_order_logits[0] if high_order_logits is not None else 0
        with tf.name_scope('NoveltyDetector') as global_scope:
            # self.score = []
            # self.output = []
            first_layer_output = []
            self.num_weights = 0
            self.num_effective_weights = 0
            # for classifier_num in range(num_of_classifiers):
            weights = []
            biases = []
            layers = [features_vect]
            layers_widths_ = [int(features_vect.shape[-1])] + layers_widths
            # for layer_num in range(1 + conv_logits):
            with tf.variable_scope('layer_0') as scope:
                # if layer_num < len(layers_widths):
                weights.append(tf.Variable(tf.truncated_normal([int(layers_widths_[0]),int(layers_widths_[1])],stddev=STDDEV), name='weights'))
                biases.append(tf.Variable(tf.zeros([layers_widths_[1]]), name='biases'))
                tf.summary.histogram(weights[-1].name, weights[-1])
                tf.summary.histogram(biases[-1].name, biases[-1])
                print('FC layer[0] input shape:', layers[0].shape)
                layers.append(FC_layer(tf.reshape(layers[0], [-1, np.prod(layers[-1].get_shape().as_list()[1:])]),
                    weights[0], biases[0], keep_prob=keep_prob, name=scope.name,batch_norm=batch_norm))
                print('layer output shape:', layers[-1].shape, 'name:', scope.name)
                first_layer_output.append(layers[-1])
            added_weights = sum([np.prod(weight.get_shape().as_list()) for weight in weights]) + \
                            sum([np.prod(weight.get_shape().as_list()) for weight in biases])
            self.num_weights += added_weights
            self.num_effective_weights += sum([np.prod([int(weight.get_shape().as_list()[0] * high_order_logits / logits_shape[1]),
                weight.get_shape().as_list()[1]]) for weight in weights]) + sum([np.prod(weight.get_shape().as_list()) for weight in biases])
            weights = []
            biases = []
            layers[-1] = tf.stack(first_layer_output, axis=2)
            layers[-1] = tf.reshape(layers[-1], [-1, int(layers_widths_[1])])
            for layer_num in range(1, len(layers_widths_)):
                with tf.variable_scope('layer' + '_%d' % (layer_num)) as scope:
                    if layer_num < len(layers_widths):
                        weights.append(tf.Variable(tf.truncated_normal([int(layers_widths_[layer_num]),
                                                                        int(layers_widths_[layer_num + 1])],
                                                                       stddev=STDDEV), name='weights'))
                        biases.append(tf.Variable(tf.zeros([layers_widths_[layer_num + 1]]), name='biases'))
                        tf.summary.histogram(weights[-1].name, weights[-1])
                        tf.summary.histogram(biases[-1].name, biases[-1])
                        print('FC layer[%d] input shape:' % (layer_num), layers[layer_num].shape)
                        layers.append(
                            FC_layer(tf.reshape(layers[layer_num], [-1, np.prod(layers[-1].get_shape().as_list()[1:])]),
                                     weights[-1], biases[-1], keep_prob=keep_prob, name=scope.name,
                                     batch_norm=batch_norm))
                        print('layer output shape:', layers[-1].shape, 'name:', scope.name)
            with tf.variable_scope('sigmoid') as scope:
                weights.append(tf.Variable(tf.truncated_normal(
                    [int(layers_widths_[-1]), 1], stddev=STDDEV), name='weights'))
                biases.append(tf.Variable(tf.zeros([1]), name='biases'))
                tf.summary.histogram(weights[-1].name, weights[-1])
                tf.summary.histogram(biases[-1].name, biases[-1])
                self.score = (tf.reshape(tf.nn.bias_add(tf.matmul(layers[-1], weights[-1]), biases[-1]), [-1]))
                layers.append(tf.sigmoid(self.score, name=scope.name))
                print('layer shape:', layers[-1].shape, 'name:', scope.name, 'dropout:', keep_prob)
            self.output = tf.reshape(layers[-1], [-1])
            added_weights = sum([np.prod(weight.get_shape().as_list()) for weight in weights]) + \
                            sum([np.prod(weight.get_shape().as_list()) for weight in biases])
            self.num_weights += added_weights
            self.num_effective_weights += added_weights
        if high_order_logits == 0:
            self.num_effective_weights = self.num_weights

    def output(self):
        return self.output

    def score(self):
        return self.score


