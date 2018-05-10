import argparse
import numpy as np
import os
import tensorflow as tf
import sys

import sklearn.metrics as metrics
import matplotlib
import time
from time import gmtime, strftime
import copy
import scipy.stats as stats
import csv
import ED_utils
from scipy.io import savemat

if 'ybahat/PycharmProjects' in os.getcwd():
    matplotlib.use('tkagg')  # I assume running in interactive mode
    CHECKPOINTS_DIR = '../Checkpoints'
    CLASSIFIER_MODELS_DIR = os.path.expanduser('../GradClassifiers')
    SUMMARY_DIR = os.path.expanduser('../Summaries_GradClassifiers')
else:
    matplotlib.use('agg')  # I assume running in Non-interactive mode
    CHECKPOINTS_DIR = os.path.expanduser('/share/data/vision-greg2/users/ybahat/Checkpoints')
    CLASSIFIER_MODELS_DIR = os.path.expanduser('/share/data/vision-greg2/users/ybahat/GradClassifiers')
    SUMMARY_DIR = os.path.expanduser('/share/data/vision-greg2/users/ybahat/Summaries_GradClassifiers')
    PERFORMANCE_TEMPLATE_FOLDER = os.path.expanduser('/home-nfs/ybahat/experiments/GMMonVgg16/CIFAR10_alex/Code')
import matplotlib.pylab as plt

MAX_RUN_TIME = 3.7 * 60 * 60
LOGIT_4_ZERO_PROB = -30000.12345
NumOfValIters = 100  # For memory resources reasons
MIN_LR = 0.00062
AUGMENTED_INPUT = False
# LAYERS_WIDTHS = []  # []
# BATCH_SIZE = 32
NUM_EPOCHS = 4000
MIN_STEPS_NUM = 100000000
# INITIAL_LR = 0.005
STDDEV = 0.005  # 0.01#0.05
LEARNING_RATE_DECAY_FACTOR = 0.5
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
DROPOUT = False
STEPS_PER_DISPLAY_STAGE = 1000  # 5000
LOSS_AV_WIN_4_LR_DECAY = 40000
# NORMALIZATION = 'input'
SQUARE_LOSS = True
# BALANCE_CLASSES = False
# DETECTOR_TRAIN_SET = False
RESUMING_GRACE_STEPS = LOSS_AV_WIN_4_LR_DECAY
# NUM_HIGH_ENER_LOGITS = 10 #When using preLable and HighEnerLogits args, how many logits are saved

parser = argparse.ArgumentParser()
# parser.add_argument("-NNDir", default='Original', type=str, help="Folder name where the saved logits file resides.",
#                     nargs='*')
parser.add_argument("-transformations", type=str, help="Type of features to use (usually type of perturbations)", nargs=1)
# parser.add_argument("-imagesDir", type=str, help="Identifier of the images used for training the detector", nargs=1)
# parser.add_argument("-normalization", type=str, choices=['none', 'batch', 'input', 'batchInput'],
#                     help="Type of detector normalization applied", nargs='*')
parser.add_argument("-batch_size", type=int, default=32,help="Batch size for detector training", nargs=1)
parser.add_argument("-layers_widths", type=str, help="Layers widths", nargs='*')
parser.add_argument("-lr", type=float, default=0.005,help="Initial Learning Rate", nargs=1)
parser.add_argument("-train", action='store_true', help="Train the model (Don't just evaluate a trained model)")
parser.add_argument("-dropout", type=float,default=0.5, help="Use drop out")
parser.add_argument("-class_balance", action='store_true',default=True, help="Class Balanced loss")
# parser.add_argument("-DTS", action='store_true',
#                     help="Use validation-set-based Training-Set instead of original training set for training")
# parser.add_argument("-VTS", action='store_true',
#                     help="Use the entire validation-set as training-Set instead of original training set for training. Allowed ONLY when training for novelty detection")
# parser.add_argument("-errorIs", type=str, choices=['novel', 'normal', 'ignored'], default=['novel'],
#                     help="For novelty detection scenario, how to treat false detection", nargs='*')
# parser.add_argument("-ConvLogits", action='store_true',
#                     help="Make the first detector layer a convolution layer")  # In this case the kernel for first
# parser.add_argument("-softmax", action='store_true', help="Apply softmax to logits feature vectors as pre-processing")
# parser.add_argument("-top5", action='store_true', help="Work in top5 scenario")
# parser.add_argument("-numLogits", type=int, help="Use files saved with only numLogits top logits per copy.")
# layer will be of size (num_image_perturbations+1)x n, where the first number in Lwidths will be W.n
# (W stands for the number of channels (width) of this first layer)
parser.add_argument("-num_scores_kept", type=int, nargs=1,default=-1,
                    help="If positive, use only the HighEnerLogits logits holding most energy, calculated per-label over training set")
# parser.add_argument("-CEloss", action='store_true', help="Use Cross-Entropy loss instead of square loss (default)")
parser.add_argument("-augment", action='store_true', help="Use training set with random image distortions", nargs='?')
# parser.add_argument("-onlyLabel", type=int, help="Train a detector for images classified as this label only", nargs='?')
# parser.add_argument("-perLabel", action='store_true',
#                     help="Train per-classified-label classifiers")  # Like the previous, but training all detectors together
# parser.add_argument("-orderPerLabel", action='store_true',
#                     help="Logits re-ordering will be done per predicted-label, rather for each sample separately")  # Similar to the previous, without separate detectors
# parser.add_argument("-desc", type=str, help="Additional description to model name", nargs='*')
# parser.add_argument("-normThrsh2display", type=float,
#                     help="During evaluation, also display threshold values corresponding to this precision rate")
# parser.add_argument("-saveNormal", type=float,
#                     help="During evaluation, save images whose novelty score is below this value for future training",
#                     nargs='*')
# parser.add_argument("-symmetricTrans", action='store_true',
#                     help="Use symmetric transformations at evaluation to improve classifier accuracy")  # Similar to the previous, without separate detectors
# parser.add_argument("-evalDir", type=str,
#                     help="Folder name where the saved logits file resides - for detector evaluation", nargs='*')
# parser.add_argument("-evalImages", type=str,
#                     help="Identifier of the images used for training the detector - for detector evaluation", nargs='*')
args = parser.parse_args()


def ConvertDirName2ClassNums(dirName):
    if type(dirName) == list:
        if dirName[0][:len('Exc_')] != 'Exc_':
            return dirName[0]
        common_classes = [label for label in CIFAR10_labels.get_label_num(dirName[0].split('_')) if type(label) == int]
        non_common = common_classes
        for name in dirName[1:]:
            cur_class = CIFAR10_labels.get_label_num(name.split('_'))
            common_classes = [label for label in common_classes if label in cur_class]
            non_common = non_common + cur_class
        non_common = [str(label) for label in set(non_common) if (label not in common_classes and type(label) == int)]
        common_classes = [str(label) for label in common_classes]
        if len(common_classes) < 1:
            raise Exception('No common excluded labels')
        non_common.sort()
        common_classes.sort()
        return 'Exc_' + '_'.join(common_classes) + ('__' + '_'.join(non_common) if len(non_common) > 0 else '')
    else:
        return '_'.join([str(label) for label in CIFAR10_labels.get_label_num(dirName.split('_'))])


if True:  # Arguments processin:
    FEATURES_CONFIG = args.transformations[0] if args.transformations is not None else 'augmentedLogitsNone'
    unlabeled_data = False
    trainOrFutureSavingImages = args.imagesDir[0]
    augmentValue4trainingOrFuture = args.augment
    # if args.saveNormal is not None:
    #     if args.train or args.evalImages is None:
    #         raise Exception(
    #             'In order to save future classifier training examples, the detector needs to be EVALUATED on a different training set')
    #     if args.evalImages[0] == 'unlabeled':
    #         unlabeled_data = True
    #     else:
    #         trainOrFutureSavingImages = args.evalImages[0]
    #     augmentValue4trainingOrFuture = None
    if args.evalDir is None:
        evaluation_dir = args.NNDir
    else:
        # if args.train:
        #     raise Exception('Not supporting a different cifar10 classifier NN for evaluation during train phase')
        evaluation_dir = args.evalDir

    excluded_labels_string = '' if args.NNDir[0].find('_Ex') == -1 else args.NNDir[0][
                                                                        args.NNDir[0].find('_Ex') + len('_Ex'):]
    model_novel_classes = [int(i) for i in excluded_labels_string]
    num_of_model_novel_classes = len(model_novel_classes)
    print('%d labels excluded' % (num_of_model_novel_classes))
    # assert not (
    # args.errorIs[0] != 'novel' and num_of_model_novel_classes == 0 and args.train and args.evalImages[0] not in ['SVHN',
                                                                                                                 'cifar100']), 'Novelty detection training scenario flag when there are no novel labels'

    if args.evalImages is None or unlabeled_data:
        images_eval_dir = args.imagesDir[0]
    else:
        images_eval_dir = args.evalImages[0]

    batch_size_validation = int(
        np.ceil(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / NumOfValIters))  # batch size for everything but training
    # Changing default parameters:
    if args.imagesDir is not None:
        print('Taking images from modified CIFAR10 DB: %s' % (trainOrFutureSavingImages))
        extra_params_string = '_m' + ConvertDirName2ClassNums(args.NNDir) + '_t' + ConvertDirName2ClassNums(
            args.imagesDir[0])
    # if args.batch_size is not None:
    batch_size = args.batch_size
    #     extra_params_string = extra_params_string + '_BSZ%d' % (BATCH_SIZE)
    # if args.LRinit is not None:
    #     INITIAL_LR = args.LRinit[0]
    #     extra_params_string = extra_params_string + '_initLR%.0e' % (INITIAL_LR)
    # if args.Lwidths is not None:
    #     if len(args.Lwidths) > 0:
    #         LAYERS_WIDTHS = args.Lwidths
    #         extra_params_string = extra_params_string + '_widths' + '>'.join(
    #             [str(wid).replace('.', '_') for wid in LAYERS_WIDTHS])
    # if args.normalization is not None:
    #     NORMALIZATION = args.normalization[0]
    #     if NORMALIZATION is not 'none':
    #         extra_params_string = extra_params_string + '_' + NORMALIZATION + 'Norm'
    # if args.DO is not None:
    #     # DROPOUT = True
    #     dropout_keep_prob = float(args.DO) * 0.1
    #     extra_params_string = extra_params_string + '_DO%d' % (args.DO)
    # else:
    #     dropout_keep_prob = 1
    # if args.CEloss:
    #     SQUARE_LOSS = False
    #     extra_params_string = extra_params_string + '_CEloss'
    # if args.CB:
    #     BALANCE_CLASSES = True
    #     extra_params_string = extra_params_string + '_CB'
    if args.ConvLogits:
        extra_params_string = extra_params_string + '_ConvL'
    # if args.perLabel:
    #     extra_params_string = extra_params_string + '_perLabel'
    # if args.orderPerLabel:
    #     order_per_label = True
    #     extra_params_string = extra_params_string + '_OPL'
    # else:
    # order_per_label = False
    # order_per_label = False
    # assert args.perLabel + args.orderPerLabel <= 1, 'Cannot use the perLabel and orderPerLabel options together'
    # assert args.DTS+args.VTS<=1,'Can use either flags, not both'
    # if args.HighEnerLogits is not None:
    #     # assert args.perLabel,'HighEnerLogits is only applicable when using a per-predicted-label detector'
    #     extra_params_string = extra_params_string + '_HEL%d' % (args.HighEnerLogits[0])
    # HighEnerLogits = args.HighEnerLogits
    num_scores_kept = args.num_scores_kept
    detector_train_set = 'train'
    # if args.errorIs[0] == 'ignored':
    #     if args.train or num_of_model_novel_classes > 0:
    #         extra_params_string = extra_params_string + '_falseIgnored'
    # elif args.errorIs[0] == 'normal':
    #     if args.train or num_of_model_novel_classes > 0:
    #         extra_params_string = extra_params_string + '_falseNormal'
    if args.DTS:
        detector_train_set = 'half_validation'
        extra_params_string = extra_params_string + '_DTS'
    if args.VTS:
        assert (
               args.evalDir is not None) or unlabeled_data, 'WARNING: Training on validation set in what seems to not be a novelty detection or unlabeled scenario'
        if args.DTS:
            detector_train_set = 'novelty_half'
        else:
            detector_train_set = 'novelty_full'
        extra_params_string = extra_params_string + '_VTS'
    if args.top5:
        extra_params_string = extra_params_string + '_top5'
    if args.softmax:
        extra_params_string = extra_params_string + '_SM'
    # if args.numLogits:
    #     assert not (args.orderPerLabel or args.perLabel), 'The order of logits is already lost'
    #     extra_params_string = extra_params_string + '_NL'
    if args.augment:
        AUGMENTED_INPUT = True
        extra_params_string = extra_params_string + '_AUGM%d' % (args.augment)
    # if args.onlyLabel is not None:
    #     extra_params_string = extra_params_string + '_only%d' % (args.onlyLabel)
    # if args.desc is not None:
    #     extra_params_string = extra_params_string + '_' + args.desc[0]
    # if args.symmetricTrans:
    #     assert not args.train, 'Can only be used with a pre-trained detector'
    #     assert FEATURES_CONFIG == 'augmentedLogits0', 'Currently only supporting horizontal flip with symmetricTrans flag'

model_name = 'Grads_FCNN_' + FEATURES_CONFIG + extra_params_string


# def soft_max(logits, temperature=1):
#     soft_max = np.minimum(np.finfo(np.float32).max,np.exp(logits/temperature))
#     soft_max = soft_max/np.minimum(np.finfo(np.float32).max,np.sum(soft_max,axis=1,keepdims=True))
#     soft_max = soft_max/np.sum(soft_max,axis=1,keepdims=True)
#     return soft_max
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


def ReturnAllPairsBut(num_classes, excluded_labels):
    all_labels = sorted([i for i in range(num_classes) if i not in excluded_labels])
    pairs = [[[j, i] for j in range(all_labels[0], i) if j not in excluded_labels] for i in all_labels]
    all_pairs = []
    for pairs_list in pairs:
        for pair in pairs_list:
            all_pairs.append(pair)
    return all_pairs


def LoadGradsData(model_dir, training_set, images_dir, num_of_model_novel_classes, augmented_input=False,
                  onlyLabel=None, dont_zero=False):
    # Loading saved logits for detector training and validation
    NUM_CLASSES_4_ExLabels = 10
    features_config = FEATURES_CONFIG
    temp_dict = {'grads_true_predicted': [], 'grads_false_predicted': [], 'grads_novel_images': []}
    for model in model_dir:
        excluded_labels = None if model.find('_Ex') == -1 else model[model.find('_Ex') + len('_Ex'):]
        if excluded_labels is not None:
            excluded_labels = np.array([int(i) for i in excluded_labels])
        model_file_name = 'Gradients_' + features_config + '_%s%s' % (
        'top5_' if args.top5 else 'unsorted_' if args.numLogits is None else '',
        '' if args.numLogits is None else '%dlogits_' % (args.numLogits)) + \
                          training_set + ('_AUGM%d' % (augmented_input) if augmented_input else '') + '_' + \
                          images_dir + '.npz'
        print('Loading file ', os.path.join(CHECKPOINTS_DIR, model, model_file_name))
        grads = np.load(os.path.expanduser(os.path.join(CHECKPOINTS_DIR, model, model_file_name)))
        features_legend = grads['features_legend'][()]
        # print('features_legend:',features_legend)
        for key in grads.files:
            if key in temp_dict.keys():
                cur_features = grads[key]
                if excluded_labels is not None and not dont_zero:
                    cur_features = np.reshape(cur_features, [-1, int(cur_features.shape[1] / NUM_CLASSES_4_ExLabels),
                                                             NUM_CLASSES_4_ExLabels])
                    cur_features[:, :,
                    excluded_labels] = LOGIT_4_ZERO_PROB  # Setting the logit to this value because exp(-1000)=0 here.
                    cur_features = np.reshape(cur_features, grads[key].shape)
                temp_dict[key].append(cur_features)
                if np.any(np.isnan(temp_dict[key][-1])):
                    print(key, np.sum(np.isnan(temp_dict[key][-1])))
                    raise Exception
        # if onlyLabel is not None:  # When training a detector only for images predicted to be onlyLabel
        #     if 'grads_true_labels' in grads.files:
        #         grads_true_labels = grads['grads_true_labels']
        #         grads_false_predicted_labels = grads['grads_false_predicted_labels']
        #     else:
        #         num_classes = int(temp_dict['grads_true_predicted'][-1].shape[1] / (
        #         len(features_config[len('augmentedLogits'):].split('_')) + 1))
        #         grads_true_labels = np.argmax(temp_dict['grads_true_predicted'][-1][:, :num_classes], axis=1)
        #         grads_false_predicted_labels = np.argmax(temp_dict['grads_false_predicted'][-1][:, :num_classes],
        #                                                  axis=1)
        if num_of_model_novel_classes == 0:  # When the classifier in NNDir was trained on all 10 labels:(related to novelty detection)
            temp_dict['grads_novel_images'][-1] = temp_dict['grads_novel_images'][-1].reshape(
                [-1] + list(temp_dict['grads_true_predicted'][-1].shape[1:]))
        # if onlyLabel is not None:
        #     print('Removing all instances except for those labled %d' % (onlyLabel))
        #     temp_dict['grads_true_predicted'][-1] = temp_dict['grads_true_predicted'][-1][
        #         grads_true_labels == onlyLabel, ...]
        #     temp_dict['grads_false_predicted'][-1] = temp_dict['grads_false_predicted'][-1][
        #         grads_false_predicted_labels == onlyLabel, ...]
        valid_features = np.arange(temp_dict['grads_true_predicted'][-1].shape[1])
    for key in temp_dict.keys():
        if len(temp_dict[key]) > 0:
            temp_dict[key] = np.concatenate(temp_dict[key], axis=0)

    if temp_dict['grads_false_predicted'].size == 0:
        temp_dict['grads_false_predicted'] = np.reshape(temp_dict['grads_false_predicted'],
                                                        temp_dict['grads_novel_images'].shape)
    # if training_set in ['val', 'val_full'] and args.symmetricTrans:
    #     return temp_dict['grads_true_predicted'][:, valid_features], temp_dict['grads_false_predicted'][:,
    #                                                                  valid_features], temp_dict['grads_novel_images'][:,
    #                                                                                   valid_features], grads[
    #                'GT_labels']
    # else:
    return temp_dict['grads_true_predicted'][:, valid_features], temp_dict['grads_false_predicted'][:,
                                                                 valid_features], temp_dict['grads_novel_images'][:,
                                                                                  valid_features]


class Detector_NN:
    def __init__(self, features_vect, layers_widths, keep_prob, batch_norm, logits_shape, predicted_labels=None,
                 conv_logits=False, high_order_logits=None):
        high_order_logits = high_order_logits[0] if high_order_logits is not None else 0
        if conv_logits:
            features_vect = tf.reshape(features_vect, [-1, logits_shape[0], logits_shape[1], 1])
            assert '.' in layers_widths[
                0], 'First layer width should include the convolutional layer''s span, after the decimal point'
            conv_layer_span = int(layers_widths[0][layers_widths[0].find('.') + 1:])
            assert not (
            conv_layer_span > 1 and high_order_logits != 0), 'Makes no sense to use convolutional layer when zeroing some inputs'
            layers_widths[0] = int(np.round(float(layers_widths[0])))
        num_of_classifiers = 1 if predicted_labels is None else int(predicted_labels.get_shape()[1])
        with tf.name_scope('NoveltyDetector') as global_scope:
            # self.score = []
            # self.output = []
            first_layer_output = []
            self.num_weights = 0
            self.num_effective_weights = 0
            for classifier_num in range(num_of_classifiers):
                weights = []
                biases = []
                layers = [features_vect]
                layers_widths_ = [int(features_vect.shape[-1])] + layers_widths
                for layer_num in range(1 + conv_logits):
                    with tf.variable_scope('layer' + '_%d' % (classifier_num) + '_%d' % (layer_num)) as scope:
                        if layer_num < len(layers_widths):
                            if conv_logits:
                                if layer_num == 0:
                                    weights.append(tf.Variable(tf.truncated_normal(
                                        [logits_shape[0], conv_layer_span, layers_widths_[layer_num],
                                         layers_widths_[layer_num + 1]], stddev=STDDEV), name='weights'))
                                elif layer_num == 1:
                                    layers_widths_[layer_num] = int(np.prod(np.array(layers[-1].get_shape())[-2:]))
                                    layers[layer_num] = tf.reshape(layers[layer_num],
                                                                   [-1, np.prod(layers[-1].get_shape().as_list()[-2:])])
                            if not (conv_logits and layer_num == 0):
                                weights.append(tf.Variable(tf.truncated_normal([int(layers_widths_[layer_num]),
                                                                                int(layers_widths_[layer_num + 1])],
                                                                               stddev=STDDEV), name='weights'))
                            biases.append(tf.Variable(tf.zeros([layers_widths_[layer_num + 1]]), name='biases'))
                            tf.summary.histogram(weights[-1].name, weights[-1])
                            tf.summary.histogram(biases[-1].name, biases[-1])
                            if layer_num == 0 and conv_logits:
                                print('conv layer[%d] input shape:' % (layer_num), layers[layer_num].shape)
                                layers.append(Conv_layer(layers[layer_num], weights[layer_num], biases[layer_num],
                                                         keep_prob=keep_prob, name=scope.name, batch_norm=batch_norm,
                                                         conv_logits=True))
                            else:
                                print('FC layer[%d] input shape:' % (layer_num), layers[layer_num].shape)
                                layers.append(FC_layer(
                                    tf.reshape(layers[layer_num], [-1, np.prod(layers[-1].get_shape().as_list()[1:])]),
                                    weights[layer_num], biases[layer_num], keep_prob=keep_prob, name=scope.name,
                                    batch_norm=batch_norm))
                            print('layer output shape:', layers[-1].shape, 'name:', scope.name)
                            if layer_num == conv_logits:
                                first_layer_output.append(layers[-1])
                added_weights = sum([np.prod(weight.get_shape().as_list()) for weight in weights]) + \
                                sum([np.prod(weight.get_shape().as_list()) for weight in biases])
                self.num_weights += added_weights
                if layer_num == conv_logits:  # If in layer after convolutional (or first layer when no convolutional):
                    self.num_effective_weights += \
                        sum([np.prod([int(weight.get_shape().as_list()[0] * high_order_logits / logits_shape[1]),
                                      weight.get_shape().as_list()[1]])
                             for weight in weights]) + sum([np.prod(weight.get_shape().as_list()) for weight in biases])
                else:
                    self.num_effective_weights += added_weights
            weights = []
            biases = []
            layers[-1] = tf.stack(first_layer_output, axis=2)
            if predicted_labels is not None:
                layers[-1] = tf.reduce_sum(
                    tf.multiply(layers[-1], tf.reshape(predicted_labels, [-1, 1, num_of_classifiers])), axis=2)
            else:
                layers[-1] = tf.reshape(layers[-1], [-1, int(layers_widths_[layer_num + 1])])
            for layer_num in range(1 + conv_logits, len(layers_widths_)):
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


def batchLossAndPredictions(data, labels, sess, loss, prediction, batch_size, batch_norm_training,
                            predicted_label=None):
    num_iter_val = int(np.ceil(data.shape[0] / batch_size))
    epoch_train_loss = []
    prediction_train = []
    for iter_num in range(num_iter_val):
        cur_indexes_val = np.arange(iter_num * batch_size, min(data.shape[0], (iter_num + 1) * batch_size))
        if predicted_label is None:
            try:
                epoch_train_loss_cur, prediction_train_cur = sess.run([loss, prediction],
                                                                      feed_dict={
                                                                          features_vects: data[cur_indexes_val, :],
                                                                          GT_novelty: labels[cur_indexes_val],
                                                                          keep_prob: 1,
                                                                          batch_norm: batch_norm_training})
            except e as Exception:
                print(iter_num / num_iter_val)
                print(e)
                raise e
        else:
            epoch_train_loss_cur, prediction_train_cur = sess.run([loss, prediction],
                                                                  feed_dict={features_vects: data[cur_indexes_val, :],
                                                                             GT_novelty: labels[cur_indexes_val],
                                                                             keep_prob: 1,
                                                                             predicted_labels: predicted_label[
                                                                                               cur_indexes_val, :],
                                                                             batch_norm: batch_norm_training})
        epoch_train_loss.append(epoch_train_loss_cur)
        prediction_train.append(prediction_train_cur)
    epoch_train_loss = np.mean(epoch_train_loss)
    prediction_train = np.concatenate(prediction_train, axis=0).reshape([-1])
    return epoch_train_loss, prediction_train


def ReformatResults(data_dict):
    for key in data_dict.keys():
        if key == 'transformations':
            if data_dict[key] is None:
                data_dict[key] = ''
            else:
                data_dict[key] = '-'.join(data_dict[key][0][len('augmentedLogits'):].split('_'))
        if isinstance(data_dict[key], list):
            if isinstance(data_dict[key][0], (int, float)):  # >='0' and data_dict[key][0][0]<='9':
                data_dict[key] = '-'.join([str(item) for item in data_dict[key]])
            else:
                data_dict[key] = '-'.join([item for item in data_dict[key]])
    return data_dict


def SavePerformanceFile(detector_results, args, eval_not_train=False):
    # stats_file_name = 'detectors_performences_%s'%('evaluation' if eval_not_train else 'training')
    stats_file_name = 'detectors_performances'
    fieldnames = sorted(list(detector_results.keys()))[::-1]
    data2write = vars(copy.deepcopy(args))
    fieldnames = fieldnames + sorted(list(data2write.keys()))
    data2write.update(detector_results)
    data2write = ReformatResults(data2write)
    new_performence_file = not os.path.isfile(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], stats_file_name + '.csv'))
    recreate_csv_file = False
    with open(os.path.join(PERFORMANCE_TEMPLATE_FOLDER, 'performances_template.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        template_field_names = reader.fieldnames
        fields2add = sorted([field for field in fieldnames if field not in template_field_names])
        recreate_csv_file = len(fields2add) > 0
        template_field_names = template_field_names + fields2add
    if recreate_csv_file:
        print('Re-creating template file because the following fields were missing:', fields2add)
        with open(os.path.join(PERFORMANCE_TEMPLATE_FOLDER, 'performances_template.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=template_field_names)
            writer.writeheader()

    if not new_performence_file:
        with open(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], stats_file_name + '.csv'), 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            existing_field_names = reader.fieldnames
            recreate_csv_file = recreate_csv_file or len(existing_field_names) > len(template_field_names) or any(
                [existing_field_names[i] != template_field_names[i] for i in range(len(existing_field_names))]) or (
                                len(existing_field_names) < len(template_field_names))
        print('Re-creating local performance file because the following fields were missing:',
              sorted([field for field in template_field_names if field not in existing_field_names]))
    if recreate_csv_file:
        from shutil import copyfile
        copyfile(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], stats_file_name + '.csv'),
                 os.path.join(CHECKPOINTS_DIR, args.NNDir[0], stats_file_name + '_old.csv'))
        with open(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], stats_file_name + '.csv'), 'w') as csvfile_new, open(
                os.path.join(CHECKPOINTS_DIR, args.NNDir[0], stats_file_name + '_old.csv'), 'r') as csvfile_old:
            reader = csv.DictReader(csvfile_old)
            writer = csv.DictWriter(csvfile_new, fieldnames=template_field_names)
            writer.writeheader()
            for row in reader:
                writer.writerow(row)
    attempt_writing = True
    while attempt_writing:
        try:
            with open(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], stats_file_name + '.csv'), 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=template_field_names)
                if new_performence_file:
                    writer.writeheader()
                writer.writerow(data2write)
            attempt_writing = False
        except Exception as e:
            if not isinstance(e, PermissionError):
                raise e
            print('Failed to write to detectors_performences (permission error). retrying...')
            time.sleep(0.5)


# if args.symmetricTrans:
#     grads_true_predicted_val, grads_false_predicted_val, grads_novel_images_val, GT_labels = \
#         LoadGradsData(evaluation_dir, training_set=('val' if 'half' in detector_train_set else 'val_full'),
#                       images_dir=images_eval_dir,
#                       num_of_model_novel_classes=num_of_model_novel_classes, augmented_input=False)
# else:
grads_true_predicted_val, grads_false_predicted_val, grads_novel_images_val = \
    LoadGradsData(evaluation_dir, training_set=('val' if 'half' in detector_train_set else 'val_full'),
                  images_dir=images_eval_dir,
                  num_of_model_novel_classes=num_of_model_novel_classes, augmented_input=False)

num_manips = len(FEATURES_CONFIG[len('augmentedLogits'):].split('_')) if FEATURES_CONFIG != 'augmentedLogitsNone' else 0
num_classes = int(grads_true_predicted_val.shape[1] / (num_manips + 1))
if unlabeled_data:
    set2load = 'unlabeled'
elif detector_train_set == 'half_validation':
    set2load = 'detectorTrain'
elif detector_train_set == 'novelty_full':  # For the novelty detection scenarion only
    set2load = 'val_full'
elif detector_train_set == 'novelty_half':  # For the novelty detection scenarion only
    set2load = 'val'
else:
    set2load = 'train'

ckpt_dir_suffixes = ['']
if args.VTS and ((args.evalDir is not None and args.evalDir == args.NNDir) or not args.train):
    assert num_of_model_novel_classes > 0 or images_eval_dir == 'SVHN', 'Didn''t find excluded labels when using VTS flag and generally a novelty detector scenario.'
    training_models = [args.NNDir[0][:args.NNDir[0].find('_Ex') + len('_Ex')] + ''.join(
        [str(i) for i in sorted(model_novel_classes + pair)]) for pair in
                       ReturnAllPairsBut(num_classes, model_novel_classes)]
    print('Using %d classifiers trained while excluding labels %s and two other label' % (
    len(training_models), excluded_labels_string))
    if num_scores_kept<0 or num_classes - 4 < num_scores_kept:
        print('Since classifiers were trained on %d classes, discarding the rest %d logits' % (num_classes - 4, 4))
        num_scores_kept = [num_classes - 4]
else:
    training_models = args.NNDir

grads_true_predicted, grads_false_predicted, grads_novel_images = \
    LoadGradsData(training_models, set2load, images_dir=trainOrFutureSavingImages,
                  num_of_model_novel_classes=num_of_model_novel_classes,
                  augmented_input=augmentValue4trainingOrFuture if AUGMENTED_INPUT else False)
# print('features_legend:',features_legend)
# if args.errorIs[0] == 'ignored':
#     print(
#         'Ignoring falsely predicted samples in this novelty detection experiment (both in detector training & evaluation)')
#     grads_false_predicted_val = np.empty(shape=[0] + list(grads_true_predicted_val.shape[1:]))
#     grads_false_predicted = np.empty(shape=[0] + list(grads_true_predicted_val.shape[1:]))
if args.softmax:
    grads_true_predicted_val = ED_utils.soft_max(grads_true_predicted_val, num_classes=num_classes)
    grads_false_predicted_val = ED_utils.soft_max(grads_false_predicted_val, num_classes=num_classes)
    grads_novel_images_val = ED_utils.soft_max(grads_novel_images_val, num_classes=num_classes)
    grads_true_predicted = ED_utils.soft_max(grads_true_predicted, num_classes=num_classes)
    grads_false_predicted = ED_utils.soft_max(grads_false_predicted, num_classes=num_classes)
    grads_novel_images = ED_utils.soft_max(grads_novel_images, num_classes=num_classes)

data_mat_val = np.concatenate((grads_true_predicted_val, grads_false_predicted_val, grads_novel_images_val), axis=0)
data_mat = np.concatenate((grads_true_predicted, grads_false_predicted, grads_novel_images), axis=0)

# if args.symmetricTrans:
#     # GT_labels = np.concatenate((np.argmax(grads_true_predicted_val[:,:num_classes],axis=1),GT4false),axis=0)
#     data_mat_val = np.reshape(data_mat_val, [data_mat_val.shape[0], 1, 2, num_classes])
#     data_mat_val = np.concatenate((data_mat_val, data_mat_val[:, :, ::-1, :]), axis=1).reshape([-1, 2 * num_classes])
#     if args.top5:
#         predicted_label4symmetricTrans = np.argsort(data_mat_val[:, :num_classes], axis=1)[:, -5:].reshape([-1, 2, 5])
#     else:
#         predicted_label4symmetricTrans = np.argmax(data_mat_val[:, :num_classes], axis=1).reshape([-1, 2, 1])
# if not (args.orderPerLabel):
data_mat = ED_utils.MutualySortLogitsOfImageVersions(data_mat, num_classes=num_classes)
data_mat_val = ED_utils.MutualySortLogitsOfImageVersions(data_mat_val, num_classes=num_classes)

# if order_per_label:  # Training a different detector for images of each predicted label:
#     predicted_label = np.argmax(data_mat[:, :num_classes], axis=1)
#     if num_scores_kept>0 or order_per_label:
#         if os.path.exists(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], 'logits_energy_order.npz')):
#             print('Loading logits_energy_order.npz')
#             logits2wipe = np.load(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], 'logits_energy_order.npz'))[
#                 'logits2wipe']
#         else:
#             logits2wipe = np.zeros([num_classes, num_classes], dtype=np.uint16)
#             for cur_label in range(num_classes):
#                 per_predicted_posteriors = 0
#                 for copy_num in range(1):  # range(num_manips+1):
#                     # per_predicted_posteriors += np.mean(soft_max(data_mat[cur_label==predicted_label,:].reshape(
#                     #     [sum(cur_label==predicted_label),num_manips+1,num_classes])[:,copy_num,:]),axis=0)#/(num_manips+1)
#                     # per_predicted_posteriors += np.mean(data_mat[cur_label == predicted_label, :].reshape(
#                     #     [sum(cur_label == predicted_label), num_manips + 1, num_classes])[:, copy_num, :],axis=0)  # /(num_manips+1)
#                     per_predicted_posteriors += \
#                         np.mean(grads_true_predicted[cur_label == predicted_label[:grads_true_predicted.shape[0]],
#                                 :].reshape(
#                             [sum(cur_label == predicted_label[:grads_true_predicted.shape[0]]), num_manips + 1,
#                              num_classes])[:, copy_num, :], axis=0)  # /(num_manips+1)
#                 logits2wipe[cur_label, :] = np.argsort(per_predicted_posteriors)
#             np.savez(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], 'logits_energy_order.npz'), logits2wipe=logits2wipe)
#     predicted_label_val = np.argmax(data_mat_val[:, :num_classes], axis=1)
    # if args.perLabel:
    #     predicted_lab_multiplier = np.zeros([data_mat.shape[0], num_classes])
    #     predicted_lab_multiplier[np.arange(data_mat.shape[0]).reshape([-1]), predicted_label] = 1
    #     predicted_lab_multiplier_val = np.zeros([data_mat_val.shape[0], num_classes])
    #     predicted_lab_multiplier_val[np.arange(data_mat_val.shape[0]).reshape([-1]), predicted_label_val] = 1
    #     batch_size *= num_classes
    #     print('Training batch size and display interval changed from %d,%d to %d,%d' % (
    #     BATCH_SIZE / num_classes, STEPS_PER_DISPLAY_STAGE,
    #     BATCH_SIZE, int(STEPS_PER_DISPLAY_STAGE / np.sqrt(num_classes))))
    #     STEPS_PER_DISPLAY_STAGE = int(STEPS_PER_DISPLAY_STAGE / np.sqrt(num_classes))
# if not args.perLabel:
predicted_lab_multiplier, predicted_lab_multiplier_val = None, None
# if args.errorIs[0] == 'normal':
#     print('Treating falsly predicted samples as NORMAL in this experiment (both in detector training & evaluation)')
#     train_labels = np.concatenate((np.zeros(grads_false_predicted.shape[0] + grads_true_predicted.shape[0]),
#                                    np.ones(grads_novel_images.shape[0])), axis=0)
#     val_labels = np.concatenate((np.zeros(grads_false_predicted_val.shape[0] + grads_true_predicted_val.shape[0]),
#                                  np.ones(grads_novel_images_val.shape[0])), axis=0)
# else:
train_labels = np.concatenate((np.zeros(grads_true_predicted.shape[0]),
                               np.ones(grads_false_predicted.shape[0] + grads_novel_images.shape[0])), axis=0)
val_labels = np.concatenate((np.zeros(grads_true_predicted_val.shape[0]),
                             np.ones(grads_false_predicted_val.shape[0] + grads_novel_images_val.shape[0])), axis=0)
# if args.symmetricTrans:
#     val_labels = np.stack((val_labels, val_labels), axis=1).reshape([-1])
# if order_per_label:
#     for cur_label in range(num_classes):
#         new_order = (logits2wipe[cur_label, ::-1].reshape([1, -1])
#                      + np.arange(0, num_classes * (num_manips + 1), num_classes).reshape([-1, 1])).reshape([-1])
#         cur_data_mat = data_mat[cur_label == predicted_label, :]
#         data_mat[cur_label == predicted_label, :] = cur_data_mat[:, new_order]
#         cur_data_mat_val = data_mat_val[cur_label == predicted_label_val, :]
#         data_mat_val[cur_label == predicted_label_val, :] = cur_data_mat_val[:, new_order]
if args.train:
    # if args.orderPerLabel and not order_per_label:  # some cheat for a sanity check
    #     new_order = np.argsort(data_mat[:, :num_classes], axis=1)[:, ::-1]
    #     new_order = (
    #     new_order.reshape([-1, 1, num_classes]) + np.arange(0, num_classes * (num_manips + 1), num_classes).reshape(
    #         [1, -1, 1])).reshape([-1, num_classes * (num_manips + 1)])
    #     data_mat = data_mat[np.arange(data_mat.shape[0]).reshape([-1, 1]), new_order]
    #     new_order = np.argsort(data_mat_val[:, :num_classes], axis=1)[:, ::-1]
    #     new_order = (
    #         new_order.reshape([-1, 1, num_classes]) + np.arange(0, num_classes * (num_manips + 1), num_classes).reshape(
    #             [1, -1, 1])).reshape([-1, num_classes * (num_manips + 1)])
        data_mat_val = data_mat_val[np.arange(data_mat_val.shape[0]).reshape([-1, 1]), new_order]
    # if args.perLabel:  # or order_per_label:
    #     features_mean = np.zeros([num_classes, data_mat.shape[1]])
    #     features_std = np.zeros_like(features_mean)
    #     for class_num in range(num_classes):
    #         if any(predicted_label == class_num):
    #             features_mean[class_num, :] = np.mean(data_mat[predicted_label == class_num, :], axis=0)
    #             features_std[class_num, :] = np.std(data_mat[predicted_label == class_num, :], axis=0,
    #                                                 keepdims=True) + 0.000001
    # else:
    features_mean = np.mean(data_mat, axis=0).reshape([1, -1])
    features_std = np.std(data_mat, axis=0, keepdims=True) + 0.000001
else:
    normalization_params = np.load(os.path.join(CLASSIFIER_MODELS_DIR, model_name, 'normalization_params.npz'))
    print(normalization_params.files)
    features_mean = normalization_params['features_mean']
    features_std = normalization_params['features_std']
if 'normed_' in FEATURES_CONFIG:
    if 'appendImage_' in FEATURES_CONFIG:
        raise Exception('Unsupported yet')
print('Loaded %d train set instances (%d true/%d false/%d novel)' % (
data_mat.shape[0], grads_true_predicted.shape[0], grads_false_predicted.shape[0], grads_novel_images.shape[0]))
print('Loaded %d validation set instances (%d true/%d false/%d novel)' % (
data_mat_val.shape[0], grads_true_predicted_val.shape[0], grads_false_predicted_val.shape[0],
grads_novel_images_val.shape[0]))
print(features_mean.shape[1:], 'features. %.0e difference between given and calculated mean.' % (
np.mean(np.mean(data_mat, axis=0) - features_mean)))
im2save = None
spacial_features_range = None
# if NORMALIZATION in ['input', 'batchInput']:
# if args.perLabel:  # or order_per_label:
#     for class_num in range(num_classes):
#         data_mat[predicted_label == class_num, :] -= features_mean[class_num, :]
#         data_mat_val[predicted_label_val == class_num, :] -= features_mean[class_num, :]
#         if NORMALIZATION in ['input', 'batchInput']:
#             data_mat[predicted_label == class_num, :] /= features_std[class_num, :]
#             data_mat_val[predicted_label_val == class_num, :] /= features_std[class_num, :]
# else:
data_mat = data_mat - features_mean
print('%.0e difference between given and calculated VALIDATION set mean.' % (
np.mean(np.mean(data_mat_val, axis=0) - features_mean)))
data_mat_val = data_mat_val - features_mean
# if NORMALIZATION in ['input', 'batchInput']:
data_mat /= features_std
data_mat_val /= features_std
print('Mean VALIDATION normalized STD %.2f, STD of STD %.2e' % (
np.mean(np.std(data_mat_val, axis=0)), np.std(np.std(data_mat_val, axis=0))))
if num_scores_kept>0:
    print('Wiping out %d/%d logits' % (
    (num_classes - num_scores_kept) * (num_manips + 1), num_classes * (num_manips + 1)))
    # if args.perLabel:
    #     for cur_label in range(num_classes):
    #         cur_multiplier = np.ones([1, data_mat.shape[1]])
    #         cur_multiplier[:, (logits2wipe[cur_label, :num_classes - HighEnerLogits[0]].reshape([-1, 1])
    #                            + np.arange(0, num_classes * (num_manips + 1), num_classes).reshape([1, -1])).reshape(
    #             [-1])] = 0
    #         data_mat[cur_label == predicted_label, :] *= cur_multiplier
    #         data_mat_val[cur_label == predicted_label_val, :] *= cur_multiplier
    # else:
    cur_multiplier = np.ones([1, data_mat.shape[1]])
    cur_multiplier[:, (np.arange(num_scores_kept, num_classes).reshape([-1, 1])
                       + np.arange(0, num_classes * (num_manips + 1), num_classes).reshape([1, -1])).reshape(
        [-1])] = 0
    # print(cur_multiplier)
    data_mat *= cur_multiplier
    data_mat_val *= cur_multiplier

np.random.seed(0)
print('%.1f%%/%.1f%% of training/validation set is labeled novel' % (
100 * np.mean(train_labels), 100 * np.mean(val_labels)))
features_vects = tf.placeholder(tf.float32, [None] + list(data_mat.shape[1:]))
GT_novelty = tf.placeholder(tf.float32, [None])
keep_prob = tf.placeholder(tf.float32)
batch_norm = tf.placeholder(tf.bool)
# if args.perLabel:
#     predicted_labels = tf.placeholder(tf.float32, [None, num_classes])
# else:
predicted_labels = None
print('Model name: ', model_name)
print('Trained on Checkpoint: ', args.NNDir)
detector = Detector_NN(features_vects, layers_widths=args.layers_widths, keep_prob=keep_prob,
                       batch_norm=batch_norm,
                       logits_shape=[num_manips + 1, int(data_mat.shape[1] / (num_manips + 1))],
                       predicted_labels=predicted_labels, conv_logits=args.ConvLogits, high_order_logits=HighEnerLogits)
print('Detector has a total of %d trainable weights (%d effective)' % (
detector.num_weights, detector.num_effective_weights))
class_score = detector.score
if SQUARE_LOSS:
    ce_loss = tf.square(tf.subtract(GT_novelty, detector.output))
else:  # cross entropy loss
    ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(GT_novelty, tf.int32),
                                                             logits=tf.stack(
                                                                 [tf.zeros_like(detector.output), detector.output],
                                                                 axis=1))

if args.class_balance:
    desired_pos_class_freq = 0.5
    pos_class_freq = np.mean(train_labels)
    print('Balancing loss to reflect %.2f%% positive, instead of the original %.2f%%' % (
    100 * desired_pos_class_freq, 100 * pos_class_freq))
    ce_loss = tf.add(
        tf.multiply(tf.multiply((desired_pos_class_freq / pos_class_freq).astype(np.float32), GT_novelty), ce_loss),
        tf.multiply(tf.multiply(((1 - desired_pos_class_freq) / (1 - pos_class_freq)).astype(np.float32),
                                tf.subtract(1.0, GT_novelty)), ce_loss))
ce_loss = tf.reduce_mean(ce_loss, name=('square_loss' if SQUARE_LOSS else 'cross_entropy_loss'))
tf.summary.scalar('cross_entropy_loss', ce_loss)
tf.add_to_collection('losses', ce_loss)
loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

global_step_Novelty = tf.train.get_or_create_global_step()
num_batches_per_epoch = data_mat.shape[0] / BATCH_SIZE
lr = tf.placeholder(tf.float32, [])
cur_lr = args.lr
tf.summary.scalar('LR', lr)
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='NoveltyDetector')
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step_Novelty)
variables_to_restore = variable_averages.variables_to_restore(moving_avg_variables=train_vars)
variables_to_restore_exclude_globalStep = {key: variables_to_restore[key] for key in variables_to_restore.keys() if
                                           'NoveltyDetector' in key}
variables_to_restore_exclude_bn = {key: variables_to_restore[key] for key in variables_to_restore.keys() if
                                   any((name in key) for name in ['global_step', 'NoveltyDetector'])}
variables_averages_op = variable_averages.apply(tf.trainable_variables() + tf.get_collection('losses'))
averaged_loss = variable_averages.average(ce_loss)
tf.summary.scalar('loss_avg', averaged_loss)
keep_prob_ = args.dropout if args.train else 1
if args.train:
    print('keep_prob: ', keep_prob_)
    batch_size = min(batch_size, cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    print('Using batch size of %d' % (batch_size))
    # optimizer = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    # optimizer = tf.train.GradientDescentOptimizer(lr)
    grads = optimizer.compute_gradients(loss, var_list=train_vars)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_op = optimizer.apply_gradients(grads, global_step=global_step_Novelty)
    with tf.control_dependencies([opt_op]):
        train_op = tf.group(variables_averages_op)
else:
    eval_op = tf.group(variables_averages_op)
step = 0
if ((not args.train) or (tf.gfile.Exists(CLASSIFIER_MODELS_DIR + '/' + model_name))):
    sys.stdout.flush()
    ckpt = tf.train.latest_checkpoint(CLASSIFIER_MODELS_DIR + '/' + model_name)
    if not args.train and ckpt is None:
        raise Exception('Could not find detector %s' % (model_name))
    train_resume = 'True' if args.train else 'Eval'
    print('%s using checkpoint ' % ('Continue training' if args.train else 'Evaluating'), ckpt)
else:
    ckpt = None
    print('Creating new model ', model_name)
    creation_timestamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    train_resume = ''
    print('Variables initial STDDEV is %.3e' % (STDDEV))
    sys.stdout.flush()
    tf.gfile.MakeDirs(CLASSIFIER_MODELS_DIR + '/' + model_name)
    np.savez(os.path.join(CLASSIFIER_MODELS_DIR, model_name, 'normalization_params.npz'), features_mean=features_mean,
             features_std=features_std)
if im2save is not None:
    plt.imsave(os.path.expanduser(CLASSIFIER_MODELS_DIR + '/' + model_name + '/early_image'), im2save[:, :, -3:])
    plt.imsave(os.path.expanduser(CLASSIFIER_MODELS_DIR + '/' + model_name + '/early_image2'), im2save[:, :, :3])
    plt.imsave(os.path.expanduser(CLASSIFIER_MODELS_DIR + '/' + model_name + '/early_image_normed'),
               np.linalg.norm(im2save[:, :, :3], axis=2))
    plt.imsave(os.path.expanduser(CLASSIFIER_MODELS_DIR + '/' + model_name + '/early_image_normed2'),
               np.linalg.norm(im2save[:, :, -3:], axis=2))

with tf.Session() as sess:
    saver = tf.train.Saver(variables_to_restore)
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR + '/' + model_name, sess.graph)
    merged = tf.summary.merge_all()
    if ckpt is not None:
        checkpoint_reader = tf.train.NewCheckpointReader(ckpt)
        restore_bn = checkpoint_reader.has_tensor('layer_1/bn/beta') or ('batch' not in NORMALIZATION)
        print('Batch normalization params %sfound in checkpoint' % ('' if restore_bn else 'NOT '))
        if restore_bn:
            saver.restore(sess, ckpt)
        else:
            tf.train.Saver(variables_to_restore_exclude_bn).restore(sess, ckpt)
            saver.save(sess, CLASSIFIER_MODELS_DIR + '/' + model_name + '/model.ckpt-%d' % (step))
            print('re-saved the restored model with global step=1')
        old_file = np.load(os.path.expanduser(CLASSIFIER_MODELS_DIR + '/' + model_name + '/training_stats.npz'))
        loss_train_total = list(old_file['loss_train_total'])
        loss_val_total = list(old_file['loss_val_total'])
        AUC_val_total = list(old_file['AUC_val_total'])
        grace_steps_left = RESUMING_GRACE_STEPS
        if 'creation_timestamp' in old_file.files:
            creation_timestamp = old_file['creation_timestamp']
        else:
            creation_timestamp = ''
        if 'AUROC_val_total' in old_file.files:
            AUROC_val_total = list(old_file['AUROC_val_total'])
            AUROC_val_total = AUROC_val_total + list(np.nan * np.ones(len(AUC_val_total) - len(AUROC_val_total)))
        else:
            AUROC_val_total = list(np.nan * np.ones(len(AUC_val_total)))
        if 'AUC_train_total' in old_file.files:
            AUC_train_total = list(old_file['AUC_train_total'])
            AUC_train_total = AUC_train_total + list(np.nan * np.ones(len(AUC_val_total) - len(AUC_train_total)))
        else:
            AUC_train_total = list(np.nan * np.ones(len(AUC_val_total)))
        if 'latest_lr' in old_file.files:
            cur_lr = old_file['latest_lr']
            restarting_lr = old_file['latest_lr']  # For the grace period after resuming training
        if 'lr_history' in old_file.files:
            lr_history = list(old_file['lr_history'])
        else:
            lr_history = list(np.full_like(loss_val_total, fill_value=np.nan))
        if 'loss_train_latest_saved' in old_file.files:
            loss_train_latest_saved = old_file['loss_train_latest_saved']
            if not restore_bn and args.train:
                loss_train_latest_saved += 0.01
                print(
                    'Adding 1/00 to previously saved loss to encourage batch normalization parameters saving once they are back on track')
        else:
            loss_train_latest_saved = 1
        print('Restored training stats from ', CLASSIFIER_MODELS_DIR + '/' + model_name + '/training_stats.npz')
    else:
        saver.save(sess, CLASSIFIER_MODELS_DIR + '/' + model_name + '/model.ckpt-%d' % (step))
        loss_train_total = []
        loss_val_total = []
        AUC_val_total = []
        AUROC_val_total = []
        lr_history = []
        loss_train_latest_saved = 1
        grace_steps_left = 0
        AUC_train_total = []
    if args.train:
        past_steps = len(loss_train_total) * STEPS_PER_DISPLAY_STAGE
        step = past_steps
        latest_saved_step = past_steps
        steps_num = max(int(np.ceil(NUM_EPOCHS * cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size)), MIN_STEPS_NUM)
        steps_num = np.ceil(steps_num / STEPS_PER_DISPLAY_STAGE) * STEPS_PER_DISPLAY_STAGE + 1
        print('Training for %d steps (%d epochs)' % (
        steps_num - past_steps, (steps_num - past_steps) * batch_size / cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN))
        min_num_thresholds4saving = data_mat.shape[0] / 10
    else:
        # sess.run(eval_op,feed_dict = {features_vects: np.empty(shape=[0,data_mat.shape[1]]), GT_novelty: np.empty([0]), keep_prob:1,batch_norm:False});print('Running eval op')
        past_steps = 0
        steps_num = 1
    np.random.seed(0)
    reshuffles = 0
    latest_lr_update = np.nan
    start_time = time.time()
    stage_time = start_time
    first_iteration = True
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        while (step < steps_num) and not coord.should_stop():
            if step % STEPS_PER_DISPLAY_STAGE == 0 or first_iteration:
                accumulated_time = time.time() - start_time
                if first_iteration:
                    # baseline_scores_val = BaselineScores(np.concatenate((grads_true_predicted_val,grads_false_predicted_val,grads_novel_images_val),axis=0),num_classes)
                    baseline_scores_val = ED_utils.BaselineScores(
                        np.concatenate((grads_true_predicted_val, grads_false_predicted_val, grads_novel_images_val),
                                       axis=0), num_classes=num_classes, soft_max_input=args.softmax,
                        top5_baseline=False)
                    samples_invalid4baseline = np.argwhere(np.isnan(baseline_scores_val)).reshape([-1])
                    if len(samples_invalid4baseline) > 0:
                        print(
                            'Discarding %d/%d (%.3f) samples because their baseline score is invalid (probably all logits are too low)' % (
                            len(samples_invalid4baseline), data_mat_val.shape[0],
                            len(samples_invalid4baseline) / data_mat_val.shape[0]))
                        baseline_scores_val = np.delete(baseline_scores_val, samples_invalid4baseline, 0)
                        data_mat_val = np.delete(data_mat_val, samples_invalid4baseline, 0)
                        val_labels = np.delete(val_labels, samples_invalid4baseline, 0)
                        if predicted_lab_multiplier_val is not None:
                            predicted_lab_multiplier_val = np.delete(predicted_lab_multiplier_val,
                                                                     samples_invalid4baseline, 0)

                time_is_up = accumulated_time > MAX_RUN_TIME
                if time_is_up:
                    ckpt = tf.train.latest_checkpoint(CLASSIFIER_MODELS_DIR + '/' + model_name)
                    saver.restore(sess, ckpt)
                    print(
                        'In preperation to quit after time was exhausted, restoring model from checkpoint %s' % (ckpt))

                loss_val, prediction_val = batchLossAndPredictions(data_mat_val, val_labels, sess, loss, class_score,
                                                                   batch_size_validation,
                                                                   predicted_label=predicted_lab_multiplier_val,
                                                                   batch_norm_training=False)
                prediction_val = np.reshape(prediction_val, [-1])
                if np.isnan(latest_lr_update):
                    latest_lr_update = len(loss_train_total)
                area_under_curve_val = metrics.average_precision_score(y_true=val_labels, y_score=prediction_val)
                area_under_ROC_val = metrics.roc_auc_score(y_true=val_labels, y_score=prediction_val)
                area_under_curve_val_Normality = metrics.average_precision_score(y_true=1 - val_labels,
                                                                                 y_score=-1 * prediction_val)
                coverage_val, accuracy_val = ED_utils.AccuracyVsCoveragePlot(val_labels, prediction_val)
                coverage_accuracy_AUC_val = metrics.auc(coverage_val, accuracy_val)
                # bound_cal = risk_control()
                # if not first_iteration:
                #     [theta, b_star] = bound_cal.bound(rstar=0.2,delta=0.001,kappa=-1*prediction_val,residuals=val_labels)
                epoch_train_loss, prediction_train = batchLossAndPredictions(data_mat, train_labels, sess, loss,
                                                                             class_score, batch_size_validation,
                                                                             predicted_label=predicted_lab_multiplier,
                                                                             batch_norm_training=False)
                if len(set(train_labels)) > 1:
                    _, _, thresholds_train = metrics.precision_recall_curve(y_true=train_labels,
                                                                            probas_pred=prediction_train)
                    area_under_curve_train = metrics.average_precision_score(y_true=train_labels,
                                                                             y_score=prediction_train)
                    area_under_ROC_train = metrics.roc_auc_score(y_true=train_labels, y_score=prediction_train)
                    area_under_curve_train_Normality = metrics.average_precision_score(y_true=1 - train_labels,
                                                                                       y_score=-1 * prediction_train)
                else:  # Added when I used a detector trained for error on a novelty detection task with SVHN as novel, with errorIs=ignored
                    thresholds_train, area_under_curve_train, area_under_ROC_train, area_under_curve_train_Normality = np.nan * train_labels, np.nan, np.nan, np.nan
                coverage_train, accuracy_train = ED_utils.AccuracyVsCoveragePlot(train_labels, prediction_train)
                coverage_accuracy_AUC_train = metrics.auc(coverage_train, accuracy_train)
                if args.train and not time_is_up:
                    grace_steps_left -= STEPS_PER_DISPLAY_STAGE
                    AUC_train_total.append(area_under_curve_train)
                    print(
                        '(%d sec.) Step %d, LR %.2e (%d to go), %d reshuffles. Averages: Loss:%.3f (%.0e), AUC:%.3f/%.3f/%.3f. Val loss:%.3f, Val AUC:%.3f/%.3f/%.3f, Cov-Acc-AUC: (%.3f,%.3f) %d train thresholds' %
                        (time.time() - stage_time, step, cur_lr, LOSS_AV_WIN_4_LR_DECAY - step + latest_saved_step,
                         reshuffles, epoch_train_loss, epoch_train_loss - loss_train_latest_saved,
                         area_under_curve_train, area_under_curve_train_Normality, area_under_ROC_train, loss_val,
                         area_under_curve_val, area_under_curve_val_Normality, area_under_ROC_val,
                         coverage_accuracy_AUC_train, coverage_accuracy_AUC_val, thresholds_train.size))
                    stage_time = time.time()
                    loss_train_total.append(epoch_train_loss)
                    loss_val_total.append(loss_val)
                    AUC_val_total.append(area_under_curve_val)
                    AUROC_val_total.append(area_under_ROC_val)
                    lr_history.append(cur_lr)
                    if ((
                                step - latest_saved_step > LOSS_AV_WIN_4_LR_DECAY) or epoch_train_loss > 2 * loss_train_latest_saved or thresholds_train.size == 1):  # or thresholds.size<train_labels.size/10):
                        # Restoring the previously saved model after too many iterations without training loss decrease or when the training loss is more than twice the lowest one so far (Except for the case of restarting training, for RESUMING_GRACE_STEPS).
                        if cur_lr * LEARNING_RATE_DECAY_FACTOR < MIN_LR and grace_steps_left <= 0:
                            MAX_RUN_TIME = accumulated_time
                            print('Learning rate reached its lower bound. Will stop training on the next display step.')
                        else:
                            cur_lr *= LEARNING_RATE_DECAY_FACTOR
                            epoch_train_loss = loss_train_latest_saved
                            ckpt = tf.train.latest_checkpoint(CLASSIFIER_MODELS_DIR + '/' + model_name)
                            saver.restore(sess, ckpt)
                            print(
                                'Restoring model from step %d and dropping learning rate to %.3e (Grace steps left=%d)' % (
                                latest_saved_step, cur_lr, grace_steps_left))
                    if (
                            step > past_steps or step == 0) and epoch_train_loss < loss_train_latest_saved and thresholds_train.size >= min_num_thresholds4saving:  # Saving the model because the training loss is the lowest so far:
                        saver.save(sess, CLASSIFIER_MODELS_DIR + '/' + model_name + '/model.ckpt-%d' % (step))
                        loss_train_latest_saved = epoch_train_loss
                        latest_saved_step = step
                        print('Updated the detector model')
                        if grace_steps_left > 0 and cur_lr < restarting_lr:
                            cur_lr = restarting_lr
                            print('Restoring the learning rate when training was resumed (%.3e)' % (cur_lr))
                        # Saving instances numbers of especially well or badly classified positive and negative examples, for debugging:
                        num_instances_per_category = 10
                        prediction_score_order = np.argsort(prediction_val)
                        # print(np.mean(val_labels),prediction_score_order.shape,val_labels.shape)
                        TruePos = prediction_score_order[val_labels[prediction_score_order] == 1][
                                  -num_instances_per_category:]
                        TrueNeg = prediction_score_order[val_labels[prediction_score_order] == 0][
                                  :num_instances_per_category]
                        FalsePos = prediction_score_order[val_labels[prediction_score_order] == 0][
                                   -num_instances_per_category:]
                        FalseNeg = prediction_score_order[val_labels[prediction_score_order] == 1][
                                   :num_instances_per_category]
                        np.savez(CLASSIFIER_MODELS_DIR + '/' + model_name + '/extremeSampleInds.npz', TruePos=TruePos,
                                 TrueNeg=TrueNeg, FalsePos=FalsePos, FalseNeg=FalseNeg)

                    reshuffles = 0
                    sys.stdout.flush()
                    np.savez(CLASSIFIER_MODELS_DIR + '/' + model_name + '/training_stats.npz',
                             loss_train_total=loss_train_total, loss_val_total=loss_val_total,
                             AUC_val_total=AUC_val_total, AUROC_val_total=AUROC_val_total,
                             AUC_train_total=AUC_train_total, latest_lr=cur_lr,
                             loss_train_latest_saved=loss_train_latest_saved, lr_history=lr_history,
                             creation_timestamp=creation_timestamp)
                else:  # If in evaluation mode or if trained and time is up
                    find_image4fig = False
                    if find_image4fig:
                        assert len(samples_invalid4baseline) == 0, 'Image indecis are translated...'
                        rank_by_BL = np.argsort(np.argsort(-1 * baseline_scores_val[val_labels == 1]))
                        rank_by_mine = np.argsort(np.argsort(prediction_val[val_labels == 1]))
                        false_images_indecis = np.arange(val_labels.shape[0])
                        false_images_indecis = false_images_indecis[val_labels == 1]
                        # I want an image whose BL score is low and my score is high:
                        chosen_image_nums = false_images_indecis[np.argsort(-rank_by_mine + rank_by_BL)[:5]]
                        print(chosen_image_nums, val_labels.shape[0])
                        data_mat_val = np.concatenate(
                            (grads_true_predicted_val, grads_false_predicted_val, grads_novel_images_val), axis=0)
                        for chosen_image_num in chosen_image_nums:
                            print(
                                'Val image %d: BL score %.3f, my score %.3f (out of %.3f : %.3f), classified as %d' % (
                                chosen_image_num, baseline_scores_val[chosen_image_num],
                                prediction_val[chosen_image_num], np.min(prediction_val), np.max(prediction_val),
                                np.argmax(data_mat_val[chosen_image_num, :num_classes])))
                            print('logits:', data_mat_val[chosen_image_num, :].reshape([-1, num_classes]))
                        raise Exception
                    if args.train:
                        plt.figure(1)
                        plt.plot(np.log(AUROC_val_total))
                        lr_history = np.log(np.array(lr_history))
                        lr_normalizer = np.std(np.log(AUROC_val_total)) / np.std(lr_history)
                        # print(lr_normalizer,type(lr_normalizer),type(lr_history))
                        plt.plot((lr_history - np.mean(lr_history)) * lr_normalizer + np.mean(np.log(AUROC_val_total)))
                        plt.xlabel('x%d steps' % (STEPS_PER_DISPLAY_STAGE))
                        plt.gcf().set_size_inches(10, 10)
                        # plt.legend(['log(Train loss)','log(Val loss)','log(Val AUC)','log(Train_AUC)','log(Val_AUROC)','LR'],loc='best')
                        plt.legend(['log(Val_AUROC)', 'LR (manipulated values)'], loc='best')
                        plt.savefig(CLASSIFIER_MODELS_DIR + '/' + model_name + '/training_stats.png')
                    # if args.saveNormal is not None:
                    #     model_file_name = 'Gradients_' + FEATURES_CONFIG + (
                    #     '_unsorted_unlabeled' if unlabeled_data else '_train') + \
                    #                       '_' + trainOrFutureSavingImages + '.npz'
                    #     print('Re-loading file %s for producing future training examples' % (
                    #     os.path.join(CHECKPOINTS_DIR, args.NNDir[0], model_file_name)))
                    #     grads_file = np.load(
                    #         os.path.expanduser(os.path.join(CHECKPOINTS_DIR, args.NNDir[0], model_file_name)))
                    #     if unlabeled_data:
                    #         corresponding_serial_nums = grads_file['grads_true_serial_num']
                    #         corresponding_predicted_labels = grads_file['predicted_labels']
                    #         print(
                    #             'Saved %d (%.2f) indecis for future classifier training. Actual accuracy is unknow - unlabeled data.' % (
                    #             sum(prediction_train <= args.saveNormal[0]),
                    #             np.mean(prediction_train <= args.saveNormal[0])))
                    #     else:
                    #         corresponding_serial_nums = np.concatenate(
                    #             (grads_file['grads_true_serial_num'], grads_file['grads_false_serial_num']), axis=0)
                    #         corresponding_predicted_labels = np.concatenate(
                    #             (grads_file['grads_true_labels'], grads_file['grads_false_predicted_labels']), axis=0)
                    #         corresponding_correctly_classified = np.concatenate((np.ones_like(
                    #             grads_file['grads_true_serial_num']), np.zeros_like(
                    #             grads_file['grads_false_serial_num'])), axis=0)
                    #         print(
                    #             'Saved %d (%.2f) indecis for future classifier training. %.2f of them are correct.' % (
                    #             sum(prediction_train <= args.saveNormal[0]),
                    #             np.mean(prediction_train <= args.saveNormal[0]),
                    #             np.mean(corresponding_correctly_classified[prediction_train <= args.saveNormal[0]])))
                    #     np.savez(os.path.expanduser(os.path.join(CLASSIFIER_MODELS_DIR, model_name,
                    #                                              'normal%.2eFrom_' % (args.saveNormal[
                    #                                                                       0]) + trainOrFutureSavingImages + '.npz')),
                    #              **{'corresponding_serial_nums': corresponding_serial_nums[
                    #                  prediction_train <= args.saveNormal[0]],
                    #                 'corresponding_predicted_labels': corresponding_predicted_labels[
                    #                     prediction_train <= args.saveNormal[0]]})
                    if unlabeled_data:
                        sys.exit('Aborting run, because there are no GT labels')
                    print(
                        'Training set: %d thresholds. Thresholds range is (%.2e:%.2e), max applies for %.2f of samples' % (
                        thresholds_train.shape[0], min(prediction_train), max(prediction_train),
                        np.mean(prediction_train == max(prediction_train))))
                    print('Loss: (%.3f,%.3f) AUC: (%.3f,%.3f) AUROC:(%.3f,%.3f)' % (
                    epoch_train_loss, loss_val, area_under_curve_train, area_under_curve_val, area_under_ROC_train,
                    area_under_ROC_val))
                    _, _, thresholds_val = metrics.precision_recall_curve(y_true=val_labels, probas_pred=prediction_val)
                    print(
                        'Validation set: %d thresholds. Thresholds range is (%.2e:%.2e), max applies for %.2f of samples' % (
                        thresholds_val.shape[0], min(prediction_val), max(prediction_val),
                        np.mean(prediction_val == max(prediction_val))))
                    P_metric_Normality, R_metric_Normality, thresholds_Normality = metrics.precision_recall_curve(
                        y_true=-1 * (val_labels - 1), probas_pred=-1 * prediction_val)
                    # if args.normThrsh2display is not None:
                    #     print(
                    #         'Threshold values around %.3f normality precision:[%.2e,%.2e]. This is the %.2f thresholds percentile.' % (
                    #         args.normThrsh2display,
                    #         -1 * thresholds_Normality[np.less(P_metric_Normality[1:], args.normThrsh2display)][-1],
                    #         -1 * thresholds_Normality[np.greater(P_metric_Normality[1:], args.normThrsh2display)][0],
                    #         np.argwhere(np.less(P_metric_Normality[1:], args.normThrsh2display))[-1] / (
                    #         len(P_metric_Normality) - 1)))
                    if not (args.saveNormal is not None or args.normThrsh2display is not None):
                        classifier_acc = grads_true_predicted_val.shape[0] / (
                        grads_true_predicted_val.shape[0] + grads_false_predicted_val.shape[0])
                        # if args.symmetricTrans:
                        #     version_preference = np.argmin(prediction_val.reshape([-1, 2]), axis=1).reshape([-1])
                        #     predicted_using_detector = predicted_label4symmetricTrans[
                        #         np.arange(GT_labels.shape[0]).reshape([-1]), version_preference, ...]
                        #     pre_detector_correctness = np.any(
                        #         np.equal(predicted_label4symmetricTrans[:, 0, ...], GT_labels.reshape([-1, 1])), axis=1)
                        #     post_detector_correctness = np.any(
                        #         np.equal(predicted_using_detector, GT_labels.reshape([-1, 1])), axis=1)
                        #     # post_detector_correctness = predicted_using_detector==GT_labels
                        #     print('Original accuracy was %.4f, improved to %.4f' % (
                        #     classifier_acc, np.mean(post_detector_correctness)))
                        #     version_changed = version_preference == 1
                        #     prediction_changed = np.not_equal(pre_detector_correctness, post_detector_correctness)
                        #     changed_for_better = np.logical_and(pre_detector_correctness == 0,
                        #                                         post_detector_correctness == 1)
                        #     changed_for_worse = np.logical_and(pre_detector_correctness == 1,
                        #                                        post_detector_correctness == 0)
                        #     print(
                        #         '%.3f of images preffered flipped, %.3f changed prediction, %.3f of them for the better and %.3f for the worse' % (
                        #         np.mean(version_changed), np.mean(prediction_changed),
                        #         np.mean(changed_for_better[prediction_changed]),
                        #         np.mean(changed_for_worse[prediction_changed])))
                        #     raise Exception
                        detected_Normal = np.array(
                            [np.mean(-1 * prediction_val > thresh) for thresh in thresholds_Normality])
                        AUC_baseline_val = metrics.average_precision_score(y_true=val_labels,
                                                                           y_score=-1 * baseline_scores_val)
                        AUROC_baseline_val = metrics.roc_auc_score(y_true=val_labels, y_score=-1 * baseline_scores_val)
                        fpr_baseline, tpr_baseline, _ = metrics.roc_curve(y_true=val_labels,
                                                                          y_score=-1 * baseline_scores_val)
                        fpr_val, tpr_val, _ = metrics.roc_curve(y_true=val_labels, y_score=prediction_val)
                        baseline_coverage, baseline_accuracy = ED_utils.AccuracyVsCoveragePlot(val_labels,
                                                                                               -1 * baseline_scores_val)
                        AUC_CA_baseline_val = metrics.auc(baseline_coverage, baseline_accuracy)
                        AUC_baseline_val_Normality = metrics.average_precision_score(y_true=-1 * (val_labels - 1),
                                                                                     y_score=baseline_scores_val)
                        detector_results = {'step': step, 'train_resume': train_resume, 'loss_train': epoch_train_loss,
                                            'loss_val': loss_val, 'creation_timestamp': creation_timestamp,
                                            'AUC_novel_train': area_under_curve_train,
                                            'AUC_normal_train': area_under_curve_train_Normality,
                                            'AUC_ROC_train': area_under_ROC_train,
                                            'AUC_CA_train': coverage_accuracy_AUC_train,
                                            'AUC_novel_val': area_under_curve_val,
                                            'AUC_normal_val': area_under_curve_val_Normality,
                                            'AUC_ROC_val': area_under_ROC_val,
                                            'AUC_CA_val': coverage_accuracy_AUC_val,
                                            'PIRA_val': ED_utils.PostIdealRejectionAccuracy(coverage_val, accuracy_val,
                                                                                            classifier_acc),
                                            'BL_AUC_novel_val': AUC_baseline_val,
                                            'BL_AUC_normal_val': AUC_baseline_val_Normality,
                                            'BL_AUC_ROC_val': AUROC_baseline_val,
                                            'BL_AUC_CA_val': AUC_CA_baseline_val,
                                            'BL_PIRA': ED_utils.PostIdealRejectionAccuracy(baseline_coverage,
                                                                                           baseline_accuracy,
                                                                                           classifier_acc),
                                            'CLS_accuracy': classifier_acc,
                                            'Num_weights': detector.num_weights,
                                            'Num_effective_weights': detector.num_effective_weights}
                        SavePerformanceFile(detector_results, args)
                        if not args.train:
                            savemat(
                                os.path.join(CLASSIFIER_MODELS_DIR, model_name, 'curves_%s.mat' % (FEATURES_CONFIG)),
                                {'coverage_val': coverage_val, 'accuracy_val': accuracy_val,
                                 'baseline_coverage': baseline_coverage, 'baseline_accuracy': baseline_accuracy,
                                 'fpr_val': fpr_val, 'tpr_val': tpr_val, 'fpr_baseline': fpr_baseline,
                                 'tpr_baseline': tpr_baseline, 'AUROC_baseline': AUROC_baseline_val,
                                 'AUROC_val': area_under_ROC_val, 'AUCAC_baseline': AUC_CA_baseline_val,
                                 'AUCAC_val': coverage_accuracy_AUC_val})
                    print('Exiting run after %s' % (time.strftime('%H:%M:%S', time.gmtime(accumulated_time))))
                    sys.exit(0)
                first_iteration = False
            if step % np.ceil(num_batches_per_epoch) == 0:
                random_perm = np.random.permutation(data_mat.shape[0])
                # if args.perLabel:
                #     random_perm = random_perm[np.argsort(predicted_label[random_perm])]
                data_mat = data_mat[random_perm, :]
                train_labels = train_labels[random_perm]
                # if args.perLabel:
                #     predicted_label = predicted_label[random_perm]
                #     predicted_lab_multiplier = predicted_lab_multiplier[random_perm, :]
                reshuffles += 1
            if args.train:
                try:
                    cur_indexes = np.mod(np.arange(step * batch_size, (step + 1) * batch_size), data_mat.shape[0])
                    data_batch = data_mat[cur_indexes, :]
                    labels_batch = train_labels[cur_indexes]
                    cur_batch_norm = True
                    if predicted_lab_multiplier is None:
                        step, _ = sess.run([global_step_Novelty, train_op],
                                           feed_dict={features_vects: data_batch, GT_novelty: labels_batch,
                                                      keep_prob: keep_prob_, lr: cur_lr, batch_norm: cur_batch_norm})
                    else:
                        step, _ = sess.run([global_step_Novelty, train_op],
                                           feed_dict={features_vects: data_batch, GT_novelty: labels_batch,
                                                      keep_prob: keep_prob_, lr: cur_lr,
                                                      predicted_labels: predicted_lab_multiplier[cur_indexes, :],
                                                      batch_norm: cur_batch_norm})
                except:
                    print('Positive mean: ', np.mean(labels_batch))
                    raise
                # writer.add_summary(summary, step)

                if latest_saved_step > step:  # For the case of restoring an old model. latest_saved_step is initialized according to the number of saved statistics, while the latest saved model
                    # may not correspond to the latest saved statistics.
                    latest_saved_step = step
                    past_steps = step
                    # prediction = np.reshape(prediction,[-1])

    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)
    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    # plt.figure(2)
    # plt.plot(loss_train_total)
    # plt.plot(loss_val_total)
    # plt.plot(AUC_val_total)
    # plt.plot(AUC_train_total)
    # # plt.plot(recall_train_total)
    # # plt.plot(precision_train_local)
    # plt.ylim([0.0,1.0])
    # plt.xlabel('%d steps'%(STEPS_PER_DISPLAY_STAGE))
    # plt.gcf().set_size_inches(10,10)
    # plt.legend(['Train loss','Val loss','Val AUC','Train_AUC'],loc='best')
    # # plt.legend(['Train loss','Val loss','Val AUC','Train recall','Train precision'],loc='best')
    # plt.savefig(CLASSIFIER_MODELS_DIR+'/'+model_name+'/training_stats.png')