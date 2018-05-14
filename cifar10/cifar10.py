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

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import numpy as np
import numpy.matlib

from six.moves import urllib
import tensorflow as tf

import cifar10.cifar10_input as cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 132,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/share/data/vision-greg2/users/ybahat/modified_CIFAR',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
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
# INNER_DATA_DIR = 'cifar-10-batches-bin'
# INNER_DATA_DIR = 'Original0'
INNER_DATA_DIR = 'Exc_Bird_Frog'

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

def _variable_coeffs_of_sum_one(name, shape, stddev,wd=None):
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
  high_verbosity = False
  shape[0] = shape[0]-1
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  initial_value = np.matlib.repmat(np.eye(N=shape[0]+1),1,int(np.ceil(shape[1]/(shape[0]+1)))).astype(np.float32)[:shape[0],:shape[1]]
  # initial_value = 0.5*np.random.uniform(size=initial_value.shape).astype(np.float32)
  initial_value = initial_value+np.random.normal(size=initial_value.shape,scale=0.05).astype(np.float32)
  # print('initial_value: ',initial_value)
  # with tf.device('/cpu:0'):
  #     var = tf.Variable(initial_value=initial_value,name=name,expected_shape=shape,dtype=dtype)
  with tf.device('/cpu:0'):
    var = tf.Variable(tf.constant(initial_value),name=name,dtype=dtype)
  # var = _variable_on_cpu(
  #     name,
  #     shape,
  #     tf.truncated_normal_initializer(mean=0.5,stddev=stddev, dtype=dtype))
  var = tf.concat([var,1-tf.reduce_sum(var,axis=0,keep_dims=True)],axis=0)
  if wd is not None:
    if high_verbosity:
      var = tf.Print(var,[tf.reduce_min(tf.reshape(var,[-1])),tf.reduce_max(tf.reshape(var,[-1]))],'This is min(var), max(var):\n')
    var_cov = tf.matmul(var,var,transpose_b=True)
    if high_verbosity:
      var_cov = tf.Print(var_cov,[tf.reduce_min(tf.diag_part(var_cov)),tf.reduce_max(tf.diag_part(var_cov)),var_cov],'This is min(diag), max(diag), var_cov:\n')
    var_eig_vals,_ = tf.self_adjoint_eig(var_cov)
    min_eig_val_loss = tf.multiply(-wd,tf.reduce_min(var_eig_vals),name='min_eig_val_loss')
    tf.add_to_collection('losses', min_eig_val_loss)
    # limit_var_range = tf.multiply(1000.0,tf.reduce_sum(tf.reshape(tf.nn.relu(tf.abs(var)-1),[-1])),name='limit_var_range')
    # def f1():
    #   return tf.Print(limit_var_range,[limit_var_range],'Range penalty: ')
    # def f2():
    #   return limit_var_range
    # limit_var_range = tf.cond(limit_var_range>0,f1,f2)
    # tf.add_to_collection('losses', limit_var_range)
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

def _variable_with_relative_weight_decay(name, shape, stddev, wd, relative_var,relative_weights_conf=RELATIVE_WEIGHTS_CONF):
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
    new_way=True
    if relative_weights_conf=='var_mean':
      moment_axes = var.get_shape().as_list()
      moment_axes = np.arange(len(moment_axes),dtype=int)
      mean_rel,variance_rel = tf.nn.moments(relative_var,axes=moment_axes)
      mean,variance = tf.nn.moments(var,axes=moment_axes)
      weight_decay = tf.multiply(tf.square(mean_rel-mean)+tf.square(tf.multiply(4.0,variance_rel)-variance), wd, name='moments_diff_loss')
    elif relative_weights_conf=='weight_vm_bias_m':
      weights_not_biases = len(var.get_shape().as_list())==2
      mean_rel,variance_rel = tf.nn.moments(relative_var,axes=[0])
      mean,variance = tf.nn.moments(var,axes=[0])
      if weights_not_biases:
        mean_rel = tf.reduce_mean(mean_rel)
        variance_rel = tf.reduce_mean(variance_rel)
        mean = tf.reduce_mean(mean)
        variance = tf.reduce_mean(variance)
        weight_decay = tf.multiply(tf.square(mean_rel-mean)+tf.square(tf.multiply(4.0,variance_rel)-variance), wd, name='moments_diff_loss')
      else:
        weight_decay = tf.multiply(tf.square(mean_rel-mean), wd, name='moments_diff_loss')
  
    # if norm_not_diff:
    #   var_size = np.prod(np.array(shape))
    #   relative_var_size = np.prod(np.array(relative_var.get_shape().as_list()))
    #   # weight_decay = tf.multiply(tf.abs(tf.nn.l2_loss(var)/tf.cast((tf.size(var)**2)/(tf.size(relative_var)**2),tf.float32)-tf.nn.l2_loss(relative_var)), wd, name='signed_weight_loss')
    #   weight_decay = tf.multiply(tf.square(tf.nn.l2_loss(var)/tf.cast(var_size**2,tf.float32)-tf.nn.l2_loss(relative_var)/tf.cast(relative_var_size**2,tf.float32)), wd, name='signed_weight_loss')
    # else:
    #   weight_decay = tf.multiply(tf.square(tf.reduce_mean(var)-tf.reduce_mean(relative_var)), wd, name='signed_weight_loss')

    tf.add_to_collection('losses', weight_decay)
  return var

def distorted_inputs(inner_data_dir,eval_data=False,batch_size=FLAGS.batch_size,coupleWithAugmentation=None):
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
                                                  batch_size=batch_size,eval_data=eval_data,coupleWithAugmentation=coupleWithAugmentation)
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
    # conv1
    relative_restriction_weight = 1.0
    pull_up_weight = 1.0
    pull_up_ratio = 0.5
    start_from_local4 = relative_weights_conf is not None and 'local4' in relative_weights_conf

    weight_decay = 0.004

    based_on_pretrained = True if noneLabelNum<0 else False
    self.num_weights = 0
    if based_on_pretrained:
        weight_decay = 0.0
    noneLabelNum = np.abs(noneLabelNum)
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
      # reshape = tf.reshape(self.pool2, [batch_size, -1])
      # print('self.pool2 shape: ',np.prod(list(self.pool2.get_shape()[1:])))
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
      if start_from_local4:
        if based_on_pretrained:
            weights = tf.stop_gradient(weights,'fixing_pretrained_weights')
            biases = tf.stop_gradient(biases,'fixing_pretrained_biases')
        combination_coeffs = _variable_coeffs_of_sum_one(name='combination_coeffs', shape=[192,192], stddev=1/10.0,wd=relative_restriction_weight)
        local4_none = tf.matmul(self.local4,combination_coeffs,name='none')
        _activation_summary(local4_none)
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
      if noneLabelNum>0:
        print('Using relative weights configuration %s'%(relative_weights_conf))
        if relative_weights_conf=='weight_vm_bias_m' or relative_weights_conf=='var_mean':
          relative_restriction_weight = 15
          print('Adding %d "None"  outputs to the CNN'%(noneLabelNum))
          # weights = tf.concat([weights,_variable_with_relative_weight_decay('weights_none', [192, noneLabelNum],stddev=1/192.0, wd=relative_restriction_weight,relative_var=weights)],axis=1)
          # biases = tf.concat([biases,_variable_with_relative_weight_decay('biases_none',[noneLabelNum],stddev=0.0,wd=relative_restriction_weight,relative_var=biases,norm_not_diff=False)],axis=0)
          weights_none = _variable_with_relative_weight_decay('weights_none', [192, noneLabelNum],stddev=1/192.0, wd=relative_restriction_weight,relative_var=weights,relative_weights_conf=relative_weights_conf)
          if based_on_pretrained:
            print('Fixing the pretrained model')
            sys.stdout.flush()
            weights = tf.stop_gradient(weights,'fixing_pretrained_weights')
            biases = tf.stop_gradient(biases,'fixing_pretrained_biases')
          weights = tf.concat([weights,weights_none],axis=1)
          biases_none = _variable_with_relative_weight_decay('biases_none',[noneLabelNum],stddev=0.0,wd=relative_restriction_weight,relative_var=biases,relative_weights_conf=relative_weights_conf)
          biases = tf.concat([biases,biases_none],axis=0)
          _,weights_none_coeff_Var = tf.nn.moments(weights_none,axes=[0])
          tf.summary.scalar('weights_none_coeff_Var', tf.reduce_mean(weights_none_coeff_Var))
          # tf.summary.histogram('weights_none_01', weights_none[:,1])
          _,biases_none_var = tf.nn.moments(biases_none,axes=[0])
          tf.summary.scalar('biases_none_var', biases_none_var)
          # tf.summary.scalar('biases_none_01', biases_none[1])

      self.top_weights = weights
      self.top_biases = biases
      if noneLabelNum>0 and relative_weights_conf[:len('outputs_linear_comb')]=='outputs_linear_comb':
        softmax_linear_labels = tf.add(tf.matmul(self.local4, weights), biases, name=scope.name+'_labels')
        if based_on_pretrained:
            print('Fixing the pretrained model')
            sys.stdout.flush()
            softmax_linear_labels = tf.stop_gradient(softmax_linear_labels,'fixing_pretrained_model')
            weights = tf.stop_gradient(weights,'fixing_pretrained_model')
            biases = tf.stop_gradient(biases,'fixing_pretrained_model')
        if start_from_local4:
            combination_coeffs = _variable_coeffs_of_sum_one(name='combination_coeffs_top', shape=[NUM_CLASSES-1,noneLabelNum//2], stddev=1/10.0,wd=relative_restriction_weight)
            combination_coeffs_local4 = _variable_coeffs_of_sum_one(name='combination_coeffs_local4', shape=[NUM_CLASSES,noneLabelNum-noneLabelNum//2], stddev=1/10.0,wd=relative_restriction_weight)
            weights_none = tf.matmul(weights,combination_coeffs_local4,name='weights_none')
            biases_none = tf.matmul(tf.reshape(biases,[1,-1]),combination_coeffs_local4,name='biases_none')
        else:
            combination_coeffs = _variable_coeffs_of_sum_one(name='combination_coeffs', shape=[NUM_CLASSES-1,noneLabelNum], stddev=1/10.0,wd=relative_restriction_weight)
        _,combination_coeffs_Var = tf.nn.moments(combination_coeffs,axes=[1])
        tf.summary.scalar('combination_coeffs_Var', tf.reduce_mean(combination_coeffs_Var))
        self.softmax_linear  = tf.concat([softmax_linear_labels,tf.matmul(tf.concat([tf.slice(softmax_linear_labels,begin=[0,0],size=[-1,excluded_label]),
          tf.slice(softmax_linear_labels,begin=[0,excluded_label+1],size=[-1,NUM_CLASSES-1-excluded_label])],axis=1),combination_coeffs)],axis=1, name=scope.name)
        if start_from_local4:
            self.softmax_linear  = tf.concat([self.softmax_linear,tf.add(tf.matmul(local4_none,weights_none),biases_none)],axis=1,name='top_and_local4')
        if relative_weights_conf[:len('outputs_linear_comb_pull_up')]=='outputs_linear_comb_pull_up':
          logit_max = tf.reduce_max(self.softmax_linear,axis=1,keep_dims=True)
          none_max = tf.reduce_max(tf.slice(self.softmax_linear,begin=[0,NUM_CLASSES],
              size=[batch_size,noneLabelNum]),axis=1,keep_dims=True)
          pull_up_loss = tf.multiply(pull_up_weight,tf.reduce_mean(tf.square(logit_max+np.log(pull_up_ratio)-none_max)),name='pull_up_loss')
          tf.add_to_collection('losses', pull_up_loss)

      else:
        self.softmax_linear = tf.add(tf.matmul(self.local4, weights), biases, name=scope.name)
      _activation_summary(self.softmax_linear)
      self.num_weights +=  np.prod(weights.get_shape().as_list()) + np.prod(biases.get_shape().as_list())
    print('Classifier has a total of %d parameters'%(self.num_weights))

  def inference_logits(self):
    return self.softmax_linear
    # return softmax_linear
  def ReturnLogits(self):
    return self.inference_logits()
  def inference_hidden_layers(self):
    return {'pool1':self.pool1,'pool2':self.pool2,'local3':self.local3,'local4':self.local4}
  def inference_top_weights(self):
    return self.top_weights,self.top_biases


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

def sigmoid_loss(logits, labels):
  labels = tf.one_hot(labels,depth=NUM_CLASSES)
  sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_logit')
  # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  #     labels=labels, logits=logits, name='cross_entropy_per_example')
  sigmoid_cross_entropy_mean = tf.reduce_mean(sigmoid_cross_entropy, name='sigmoid_cross_entropy')
  tf.add_to_collection('losses', sigmoid_cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step,initial_lr):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  # print('step_4_lr:',step_4_lr)
  lr = tf.train.exponential_decay(initial_lr,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
  # lr = tf.Print(lr,[lr],'Current lr: ')
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    # opt = tf.train.AdamOptimizer(0.0001)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op


def maybe_download_and_extract(inner_data_dir):
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if len(inner_data_dir)>0:
    INNER_DATA_DIR = inner_data_dir
  elif not os.path.exists(os.path.join(dest_directory,INNER_DATA_DIR)):
    os.makedirs(os.path.join(dest_directory,INNER_DATA_DIR))
  if os.path.isfile(dest_directory+'/'+INNER_DATA_DIR+'/data_batch_1.bin'):
    print('Found bin files in %s, not downloading'%(dest_directory+'/'+INNER_DATA_DIR))
    return
  elif len(inner_data_dir)>0:
    raise Exception('Couldn''t find data files in %s'%(dest_directory+'/'+INNER_DATA_DIR))
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory,INNER_DATA_DIR, filename)
  if not os.path.exists(filepath):
    if len(inner_data_dir)>0:
      raise Exception('Could not find folder or files in %s'%(INNER_DATA_DIR))
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(dest_directory)
