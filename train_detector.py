import argparse
import numpy as np
import os
import tensorflow as tf
import time
import Transformations
import detector_network
import example_utils
import cifar10.cifar10 as cifar10

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Limit to 1 GPU when using an interactive session

MIN_LR = 0.005/16
LEARNING_RATE_DECAY_FACTOR = 0.5
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
TRANSFORMATIONS_LIST = ['BW','horFlip','increaseContrast3','gamma8.5','blur3']  # Transformations to be applied on images. Choose from the list of implemented
                                                                                # transformations in Transformations.py. Passing an empty list will have the
                                                                                # detector train on the logits output corresponding to the original image only.

parser = argparse.ArgumentParser()
parser.add_argument("-detector_checkpoint", type=str,default='./detector_ckpt', help="Folder name for storing the detector checkpoints")
parser.add_argument("-classifier_checkpoint", type=str, default='./cifar10/checkpoint',help="Folder containing classifier''s checkpoint")
parser.add_argument("-dataset_folder", type=str, default='./cifar10/dataset',help='Folder where dataset files reside')
parser.add_argument("-layers_widths", type=str, help="Number of channels is each of the detector''s hidden layers (and therefore number of hidden layers too)", nargs='*')
parser.add_argument("-descriptor", type=str,help="Optional string to be assigned to ROC curve and saved model")
parser.add_argument("-batch_size", type=int, default=32,help="Batch size for detector training")
parser.add_argument("-epochs", type=int, default=200,help="Maximal number of training epochs")
parser.add_argument("-test_freq", type=int, default=5,help="Test detector every how many epochs")
parser.add_argument("-figures_folder", type=str, default="./figures", help='Folder where ROC curve figures are saved',nargs=1)
parser.add_argument("-lr_decrease_epochs", type=int, default=30,help="Number of epochs without training loss drop before decreasing lr")
parser.add_argument("-lr", type=float, default=0.005,help="Initial Learning Rate")
parser.add_argument("-train_portion", type=float, default=0.5,help="Portion of validation dataset used as detector training set")
parser.add_argument("-train", action='store_true', help="Train the model (Don't just evaluate a trained model)")
parser.add_argument("-resume_train", action='store_true', help="Resume training of a pre-trained detector")
parser.add_argument("-data_normalization", action='store_true',help="Normalize feature vectors to have 0 mean and STD of 1 prior to feeding the detector")
parser.add_argument("-num_logits_per_transformation", type=int,default=-1,help="If positive, use only the HighEnerLogits logits holding most energy, calculated per-label over training set")
parser.add_argument("-no_augmentation", action='store_true', help="Avoid applying random image distortions on the detector training set (prior to performing the detector transformations)")
parser.add_argument("-L2_loss", action='store_true', help="Use L2 loss instead of cross-entropy loss")
parser.add_argument("-dropout", type=float,default=0.5, help="Keeping probability for detector training (Set 1 for not using any drop-out)")
args = parser.parse_args()
print('------------ Options -------------')
for k, v in sorted(args._get_kwargs()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

model_name = 'detector'+('_%s'%(args.descriptor) if args.descriptor is not None else '')

# Splitting the validation dataset into training and validation sets for the detector:
validation_split_filename = os.path.join(args.dataset_folder,'ValidationSetSplit_%s.npz'%(str(args.train_portion).replace('.','_')))
if os.path.exists(validation_split_filename):
    detector_train_set_indicator = np.load(validation_split_filename)['detector_train_set_indicator']
else:
    print('!!! drawing a NEW random split of the classifier''s validation set into training and validation sets for the detector !!!')
    num_of_samples = example_utils.SplitCifar10TestSet(dataset_folder=args.dataset_folder)
    detector_train_set_indicator = np.zeros([num_of_samples]).astype(np.bool)
    detector_train_set_indicator[np.random.permutation(num_of_samples)[:int(num_of_samples*args.train_portion)]] = True
    np.savez(validation_split_filename,detector_train_set_indicator=detector_train_set_indicator)
num_of_samples = example_utils.SplitCifar10TestSet(dataset_folder=args.dataset_folder,train_indicator=detector_train_set_indicator)
# Data retrieval:
data2use = tf.placeholder(dtype=tf.string)
images_stats,labels_stats = cifar10.inputs(eval_data=False, inner_data_dir=args.dataset_folder,batch_size=args.batch_size)
def StatsData():    return images_stats, labels_stats
images_val, labels_val = cifar10.inputs(eval_data=True, inner_data_dir=args.dataset_folder,batch_size=args.batch_size)
def ValData():  return images_val, labels_val
if args.no_augmentation:
    images_train, labels_train = images_stats,labels_stats
else:
    images_train,labels_train = cifar10.distorted_inputs(eval_data=False, inner_data_dir=args.dataset_folder,batch_size=args.batch_size)
def TrainData():    return images_train, labels_train
pre_transform_images, labels = tf.case({tf.equal(data2use,'stats'):StatsData,tf.equal(data2use,'train'):TrainData,tf.equal(data2use,'val'):ValData})

# Applying our transformations on the images (and handling their corresponding labels as well):
# Transformations should be applied on the raw images, before applying any standartization, whitening etc'.
transformer = Transformations.Transformer(transformations=TRANSFORMATIONS_LIST)
images,labels = transformer.TransformImages_TF_OP(pre_transform_images,labels)

# The example classifer was trained on standartized images, so applying standartization AFTER the transformations were applied:
post_processed_images = tf.map_fn(lambda im:tf.image.per_image_standardization(im),images)

train_batches_per_epoch = int(np.ceil(num_of_samples*args.train_portion / args.batch_size))
val_batches_per_epoch = int(np.ceil(num_of_samples*(1-args.train_portion) / args.batch_size))

# Example classifier model:
classifier = cifar10.inference(post_processed_images)
logits = classifier.inference_logits()
correctness = tf.nn.in_top_k(logits,labels, 1)
classifier_logits,features_vects = transformer.Process_Logits_TF_OP(logits,num_logits_per_transformation=args.num_logits_per_transformation)
classifier_softmax = tf.nn.softmax(classifier_logits)
variable_averages_C = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
variables_to_restore_C = variable_averages_C.variables_to_restore()
saver_C = tf.train.Saver(variables_to_restore_C)

# Detector labels and training place holders:
detector_labels = tf.cast(tf.logical_not(transformer.Process_NonLogits_TF_OP(correctness)),tf.float32)
keep_prob = tf.placeholder(tf.float32)
bn_learning = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32, [])

# Preparations for detector training:
train = args.train or args.resume_train
if train:
    tf.gfile.MakeDirs(os.path.join(args.detector_checkpoint, model_name))# Creating detector checkpoint folder
    # Handling training set class imbalance (correct vs. incorrect) by balancing the loss function. Calculating class imbalance:
    desired_pos_class_freq = 0.5
    with tf.Session() as sess:
        ckpt_C = tf.train.get_checkpoint_state('/home/ybahat/PycharmProjects/CFI/cifar10/checkpoint')
        # Restores from checkpoint
        sess.run(tf.global_variables_initializer())
        saver_C.restore(sess, ckpt_C.model_checkpoint_path)
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            pos_class_freq = 0
            trainset_features = []
            for batch_num in range(train_batches_per_epoch):
                if args.data_normalization:
                    # calculating training set statistics as well:
                    cur_detections,cur_features = sess.run([detector_labels,features_vects],feed_dict={data2use: 'stats', lr: 0, keep_prob: 0,bn_learning: True})
                    trainset_features.append(cur_features)
                else:
                    cur_detections = sess.run(detector_labels,feed_dict={data2use: 'stats', lr: 0, keep_prob: 0,bn_learning: True})
                pos_class_freq += np.mean(cur_detections)/train_batches_per_epoch
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
if args.data_normalization:
    if args.train:
        trainset_features = np.concatenate(trainset_features,axis=0)
        features_mean = np.mean(trainset_features,axis=0)
        features_std = np.maximum(np.std(trainset_features,axis=0),1e-3)
        np.savez(os.path.join(args.detector_checkpoint,model_name,'normalization_stats.npz'),features_mean=features_mean,features_std=features_std)
    else:
        normalization_stats = np.load(os.path.join(args.detector_checkpoint,model_name,'normalization_stats.npz'))
        features_mean = normalization_stats['features_mean']
        features_std = normalization_stats['features_std']
    features_vects = tf.divide(tf.subtract(features_vects,tf.constant(features_mean)),tf.constant(features_std))
# Creating the detector model:
detector = detector_network.Detector_NN(features_vects, layers_widths=args.layers_widths, keep_prob=keep_prob,bn_learning=bn_learning)
print('Detector has a total of %d trainable weights' % (detector.num_weights))
detector_logit = detector.logit
if train:# Detector loss function:
    print('Balancing loss to reflect %.2f%% positive, instead of the original %.2f%%' % (100 * desired_pos_class_freq, 100 * pos_class_freq))
    if not args.L2_loss:
        detector_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=detector_labels,logits=detector_logit)
    else:
        detector_loss = tf.square(tf.subtract(detector_labels, detector.output))
    detector_loss = tf.add(
        tf.multiply(tf.multiply((desired_pos_class_freq / pos_class_freq).astype(np.float32), detector_labels), detector_loss),
        tf.multiply(tf.multiply(((1 - desired_pos_class_freq) / (1 - pos_class_freq)).astype(np.float32),
                                tf.subtract(1.0, detector_labels)), detector_loss))
    detector_loss = tf.reduce_mean(detector_loss, name='detector_loss')
    tf.add_to_collection('losses',detector_loss)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

global_step_detector = tf.train.get_or_create_global_step()
cur_lr = args.lr
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Detector')
variable_averages_D = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step_detector)
variables_to_restore_D = variable_averages_D.variables_to_restore(moving_avg_variables=train_vars)
variables_averages_op = variable_averages_D.apply(tf.trainable_variables() + tf.get_collection('losses'))
keep_prob_ = args.dropout if train else 1

np.random.seed(0)
if train:
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads = optimizer.compute_gradients(loss, var_list=train_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_op = optimizer.apply_gradients(grads, global_step=global_step_detector)
    with tf.control_dependencies([opt_op]):
        train_op = tf.group(variables_averages_op)
eval_op = tf.group(variables_averages_op)

step = 0
lowest_train_loss = None
latest_best_loss = 0
prev_time = time.time()
time_per_epoch = [0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Restores classifier from checkpoint
    ckpt_C = tf.train.get_checkpoint_state('/home/ybahat/PycharmProjects/CFI/cifar10/checkpoint')
    saver_C.restore(sess, ckpt_C.model_checkpoint_path)
    saver_D = tf.train.Saver(variables_to_restore_D)# Detector saver
    if (not train) or args.resume_train:  # Restore a pre-trained detector:
        ckpt_D = tf.train.get_checkpoint_state(os.path.join(args.detector_checkpoint,model_name))
        assert ckpt_D is not None,'Could not find detector %s' % (model_name)
        print('%s using checkpoint %s' % ('Continue training' if args.resume_train else 'Evaluating', ckpt_D.model_checkpoint_path))
        saver_D.restore(sess, ckpt_D.model_checkpoint_path)
    else:# Train a new detector:
        print('Creating new model: ', model_name)
        saver_D.save(sess, os.path.join(args.detector_checkpoint,model_name,'model.ckpt-%d' % (step)))
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        for epoch in range(args.epochs):
            if epoch%args.test_freq==0:
                detector_val_logits,detector_val_labels,classifier_MSR_val,detector_val_loss = [],[],[],[]
                for batch_num in range(val_batches_per_epoch):#Evaluate detector on its validation set
                    cur_logits,cur_labels,cur_softmax,_,cur_loss = sess.run([detector_logit,detector_labels,classifier_softmax,eval_op,detector_loss],
                        feed_dict={data2use:'val',lr:cur_lr,keep_prob:1,bn_learning:False})
                    detector_val_logits.append(cur_logits)
                    detector_val_labels.append(cur_labels)
                    detector_val_loss.append(cur_loss)
                    classifier_MSR_val.append(np.max(cur_softmax,axis=1))
                # Compute area under ROC and save curves to figure:
                val_AUROC_CFI,val_AUROC_MSR = example_utils.ProcessValidationData(detector_val_logits, detector_val_labels, classifier_MSR_val,
                    figures_folder=args.figures_folder,descriptor=args.descriptor)
                detector_train_logits,detector_train_labels,classifier_MSR_train,detector_train_loss = [],[],[],[]
                for batch_num in range(train_batches_per_epoch):#Evaluate detector on its training set:
                    cur_logits, cur_labels, cur_softmax,_,cur_loss = sess.run([detector_logit, detector_labels, classifier_softmax,eval_op,detector_loss],
                        feed_dict={data2use: 'stats', lr: cur_lr, keep_prob: 1, bn_learning: False})
                    detector_train_loss.append(cur_loss)
                    detector_train_logits.append(cur_logits)
                    detector_train_labels.append(cur_labels)
                    classifier_MSR_train.append(np.max(cur_softmax, axis=1))
                train_AUROC_CFI,train_AUROC_MSR = example_utils.ProcessValidationData(detector_train_logits, detector_train_labels, classifier_MSR_train)
                print('Epoch %d (%d sec/epoch). Loss (tain/validation): (%.2e,%.2e). AUROC on (training/validation) sets: (%.3f/%.3f). AUROC using MSR: (%.3f/%.3f)'%
                      (epoch,np.mean(time_per_epoch),np.mean(detector_train_loss),np.mean(detector_val_loss),train_AUROC_CFI,val_AUROC_CFI,train_AUROC_MSR,val_AUROC_MSR))
                time_per_epoch = []
            if not train:
                print('Detector evaluation is done')
                break
            train_loss = 0
            prev_time = time.time()
            # Train the detector:
            for batch_num in range(train_batches_per_epoch):
                step, cur_loss,_ = sess.run([global_step_detector,detector_loss,train_op],feed_dict={data2use:'train',lr:cur_lr,keep_prob:keep_prob_,bn_learning:True})
                train_loss += cur_loss/train_batches_per_epoch
            time_per_epoch.append(time.time()-prev_time)
            if lowest_train_loss is None or train_loss<lowest_train_loss:
                saver_D.save(sess, os.path.join(args.detector_checkpoint, model_name, 'model.ckpt-%d' % (step)))
                lowest_train_loss = train_loss
                latest_loss_drop = epoch
            elif epoch-latest_loss_drop>args.lr_decrease_epochs:
                latest_loss_drop = epoch
                cur_lr *= LEARNING_RATE_DECAY_FACTOR
                if cur_lr<MIN_LR:
                    print('Breaking training because learning rate is below %.3e'%(MIN_LR))
                    break
                print('Dropping learning rate to %.2e after %d epochs without loss drop. Lowest training loss so far was %.4f.'%(cur_lr,args.lr_decrease_epochs,lowest_train_loss))
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)