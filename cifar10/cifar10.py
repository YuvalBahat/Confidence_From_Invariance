# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import tarfile
import numpy as np
import numpy.matlib

from six.moves import urllib
import tensorflow as tf

import cifar10.cifar10_input as cifar10_input
import sys
# if __name__ == "__main__":
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 132,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/share/data/vision-greg2/users/ybahat/modified_CIFAR',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
# The following is a solution I found here:  https://stackoverflow.com/questions/48198770/tensorflow-1-5-0-rc0-error-using-tf-app-flags
#  to a problem with the tf FLAGS I had when switching to version 1.7:
remaining_args = FLAGS([sys.argv[0]] + [flag for flag in sys.argv if flag.startswith("--")])
assert(remaining_args == [sys.argv[0]])

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

RELATIVE_WEIGHTS_CONF = 'outputs_linear_comb_pull_up'#'outputs_linear_comb'#'weight_vm_bias_m'#'var_mean''outputs_linear_comb_pull_up_local4'
# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def distorted_inputs(inner_data_dir,eval_data=False,batch_size=FLAGS.batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  # if not FLAGS.data_dir:
  #   raise ValueError('Please supply a data_dir')
  # data_dir = os.path.join(FLAGS.data_dir, inner_data_dir)
  images, labels = cifar10_input.distorted_inputs(data_dir=inner_data_dir,
                                                  batch_size=batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data,inner_data_dir,batch_size=FLAGS.batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  # if not FLAGS.data_dir:
  #   raise ValueError('Please supply a data_dir')
  # data_dir = os.path.join(FLAGS.data_dir, inner_data_dir)

  images, labels = cifar10_input.inputs(eval_data=eval_data,data_dir=inner_data_dir,batch_size=batch_size)
  # print('images shape:',images.shape,'labels shape:',labels.shape)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

class inference:
  def __init__(self,images,noneLabelNum=0,relative_weights_conf=RELATIVE_WEIGHTS_CONF,excluded_label=None,batch_size=FLAGS.batch_size,extra_FC=False):
# def inference(images):
    """Build the CIFAR-10 model.

    Args:
      images: Images returned from distorted_inputs() or inputs().

    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    weight_decay = 0.004
    self.num_weights = 0
    with tf.variable_scope('conv1') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[5, 5, 3, 64],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      self.conv1 = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(self.conv1)
      self.num_weights +=  np.prod(kernel.get_shape().as_list()) + np.prod(biases.get_shape().as_list())
    # pool1
    self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    self.norm1 = tf.nn.lrn(self.pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[5, 5, 64, 64],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(self.norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      self.conv2 = tf.nn.relu(pre_activation, name=scope.name)
      _activation_summary(self.conv2)
      self.num_weights +=  np.prod(kernel.get_shape().as_list()) + np.prod(biases.get_shape().as_list())

    # norm2
    self.norm2 = tf.nn.lrn(self.conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    self.pool2 = tf.nn.max_pool(self.norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # local3
    with tf.variable_scope('local3') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(self.pool2, shape=[-1, int(np.prod(list(self.pool2.get_shape()[1:])))])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                            stddev=0.04, wd=weight_decay)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      self.local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
      _activation_summary(self.local3)
      self.num_weights +=  np.prod(weights.get_shape().as_list()) + np.prod(biases.get_shape().as_list())

    # local4
    with tf.variable_scope('local4') as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                            stddev=0.04, wd=weight_decay)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      self.local4 = tf.nn.relu(tf.matmul(self.local3, weights) + biases, name=scope.name)
      _activation_summary(self.local4)
      self.num_weights +=  np.prod(weights.get_shape().as_list()) + np.prod(biases.get_shape().as_list())
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
      weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                            stddev=1/192.0, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
      _,weights_labels_coeff_Var = tf.nn.moments(weights,axes=[0])
      tf.summary.scalar('weights_labels_coeff_Var', tf.reduce_mean(weights_labels_coeff_Var))
      self.top_weights = weights
      self.top_biases = biases
      self.softmax_linear = tf.add(tf.matmul(self.local4, weights), biases, name=scope.name)
      _activation_summary(self.softmax_linear)
      self.num_weights +=  np.prod(weights.get_shape().as_list()) + np.prod(biases.get_shape().as_list())
    print('Classifier has a total of %d parameters'%(self.num_weights))

  def inference_logits(self):
    return self.softmax_linear
    # return softmax_linear


def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')
