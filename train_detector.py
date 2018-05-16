import argparse
import numpy as np
import os
import tensorflow as tf
# import matplotlib
import Transformations
import detector_network
import example_utils
import cifar10.cifar10 as cifar10

if 'ybahat/PycharmProjects' in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Limit to 1 GPU when using an interactive session
    # matplotlib.use('tkagg')  # I assume running in interactive mode
    CHECKPOINTS_DIR = '../Checkpoints'
    CLASSIFIER_MODELS_DIR = os.path.expanduser('../GradClassifiers')
    SUMMARY_DIR = os.path.expanduser('../Summaries_GradClassifiers')
else:
    # matplotlib.use('agg')  # I assume running in Non-interactive mode
    CHECKPOINTS_DIR = os.path.expanduser('/share/data/vision-greg2/users/ybahat/Checkpoints')
    CLASSIFIER_MODELS_DIR = os.path.expanduser('/share/data/vision-greg2/users/ybahat/GradClassifiers')
    SUMMARY_DIR = os.path.expanduser('/share/data/vision-greg2/users/ybahat/Summaries_GradClassifiers')
    PERFORMANCE_TEMPLATE_FOLDER = os.path.expanduser('/home-nfs/ybahat/experiments/GMMonVgg16/CIFAR10_alex/Code')
import matplotlib.pylab as plt

MIN_LR = 0.005/16
LEARNING_RATE_DECAY_FACTOR = 0.5
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
TRANSFORMATIONS_LIST = ['BW','horFlip','increaseContrast3','gamma8.5','blur3']

parser = argparse.ArgumentParser()
parser.add_argument("-transformations", type=str, help="Type of features to use (usually type of perturbations)", nargs=1)
parser.add_argument("-layers_widths", type=str, help="Layers widths", nargs='*')
parser.add_argument("-classifier_checkpoint", type=str, default='/home/ybahat/PycharmProjects/CFI/cifar10/checkpoint',help="Folder containing classifier''s checkpoint")
parser.add_argument("-dataset_folder", type=str, default="/home/ybahat/data/Databases/cifar10/bin", help='Folder where dataset files reside',nargs=1)
parser.add_argument("-detector_checkpoint", type=str, help="Layers widths",default='./detector_ckpt', nargs='*')
parser.add_argument("-batch_size", type=int, default=32,help="Batch size for detector training", nargs=1)
parser.add_argument("-epochs", type=int, default=1000,help="Number of training epochs", nargs=1)
parser.add_argument("-test_freq", type=int, default=5,help="Test detector every how many epochs", nargs=1)
parser.add_argument("-figures_folder", type=str, default="./figures", help='Folder where figures are saved',nargs=1)
parser.add_argument("-lr_decrease_epochs", type=int, default=30,help="Number of epochs without training loss drop before decreasing lr", nargs=1)
parser.add_argument("-lr", type=float, default=0.005,help="Initial Learning Rate", nargs=1)
parser.add_argument("-train_portion", type=float, default=0.5,help="Portion of validation dataset used as detector training set", nargs=1)
parser.add_argument("-train", action='store_true', help="Train the model (Don't just evaluate a trained model)")
parser.add_argument("-resume_train", action='store_true', help="Resume training of a pre-trained detector")
parser.add_argument("-data_normalization", action='store_true',help="Don''t use Class Balanced loss")
parser.add_argument("-num_logits_per_transformation", type=int,default=-1,help="If positive, use only the HighEnerLogits logits holding most energy, calculated per-label over training set")
parser.add_argument("-no_augmentation", action='store_true', help="Avoid using training set with random image distortions")
parser.add_argument("-dropout", type=float,default=0.5, help="Use drop out")
args = parser.parse_args()
print('------------ Options -------------')
for k, v in sorted(args._get_kwargs()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

batch_size = args.batch_size
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
images, labels = tf.case({tf.equal(data2use,'stats'):StatsData,tf.equal(data2use,'train'):TrainData,tf.equal(data2use,'val'):ValData})
transformer = Transformations.Transformer(transformations=TRANSFORMATIONS_LIST,image_size=cifar10.IMAGE_SIZE)
images,labels = transformer.TransformImages_TF_OP(images,labels)
train_batches_per_epoch = int(np.ceil(num_of_samples*args.train_portion / args.batch_size))

val_batches_per_epoch = int(np.ceil(num_of_samples*(1-args.train_portion) / args.batch_size))
# Classifier:
classifier = cifar10.inference(images)
logits = classifier.inference_logits()
correctness = tf.nn.in_top_k(logits,labels, 1)
classifier_logits,features_vects = transformer.Process_Logits_TF_OP(logits,num_logits_per_transformation=args.num_logits_per_transformation)
classifier_softmax = tf.nn.softmax(classifier_logits)
detector_labels = tf.cast(tf.logical_not(transformer.Process_NonLogits_TF_OP(correctness)),tf.float32)
variable_averages_C = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
variables_to_restore_C = variable_averages_C.variables_to_restore()
saver_C = tf.train.Saver(variables_to_restore_C)
# Detector:
keep_prob = tf.placeholder(tf.float32)
bn_learning = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32, [])

if args.train:
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
        trainset_features = np.concatenate(trainset_features,axis=0)
        features_mean = tf.constant(np.mean(trainset_features,axis=0))
        features_std = tf.constant(np.maximum(np.std(trainset_features,axis=0),1e-3))
        features_vects = tf.divide(tf.subtract(features_vects,features_mean),features_std)
    detector = detector_network.Detector_NN(features_vects, layers_widths=args.layers_widths, keep_prob=keep_prob,bn_learning=bn_learning)
    print('Detector has a total of %d trainable weights' % (detector.num_weights))
    print('Balancing loss to reflect %.2f%% positive, instead of the original %.2f%%' % (100 * desired_pos_class_freq, 100 * pos_class_freq))
    detector_logit = detector.logit
    square_loss = tf.square(tf.subtract(detector_labels, detector.output))
    square_loss = tf.add(
        tf.multiply(tf.multiply((desired_pos_class_freq / pos_class_freq).astype(np.float32), detector_labels), square_loss),
        tf.multiply(tf.multiply(((1 - desired_pos_class_freq) / (1 - pos_class_freq)).astype(np.float32),
                                tf.subtract(1.0, detector_labels)), square_loss))
square_loss = tf.reduce_mean(square_loss, name='square_loss')
tf.add_to_collection('losses',square_loss)
loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

global_step_detector = tf.train.get_or_create_global_step()
cur_lr = args.lr
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Detector')
variable_averages_D = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step_detector)
variables_to_restore_D = variable_averages_D.variables_to_restore(moving_avg_variables=train_vars)
variables_averages_op = variable_averages_D.apply(tf.trainable_variables() + tf.get_collection('losses'))
averaged_loss = variable_averages_D.average(square_loss)
keep_prob_ = args.dropout if args.train else 1
np.random.seed(0)
model_name = 'detector'
if args.train:
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    grads = optimizer.compute_gradients(loss, var_list=train_vars)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt_op = optimizer.apply_gradients(grads, global_step=global_step_detector)
    with tf.control_dependencies([opt_op]):
        train_op = tf.group(variables_averages_op)
else:
    eval_op = tf.group(variables_averages_op)

step = 0
lowest_train_loss = None
latest_best_loss = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Restores classifier from checkpoint
    ckpt_C = tf.train.get_checkpoint_state('/home/ybahat/PycharmProjects/CFI/cifar10/checkpoint')
    saver_C.restore(sess, ckpt_C.model_checkpoint_path)
    saver_D = tf.train.Saver(variables_to_restore_D)# Detector saver
    if (not args.train) or args.resume_train:  # Restore a pre-trained detector:
        ckpt_D = tf.train.get_checkpoint_state(os.path.join(args.detector_checkpoint,model_name))
        assert ckpt_D is not None,'Could not find detector %s' % (model_name)
        print('%s using checkpoint ' % ('Continue training' if args.resume_train else 'Evaluating'), ckpt_D)
        saver_D.restore(sess, ckpt_D.model_checkpoint_path)
    else:# Train a new detector:
        print('Creating new model: ', model_name)
        tf.gfile.MakeDirs(os.path.join(args.detector_checkpoint,model_name))
        saver_D.save(sess, os.path.join(args.detector_checkpoint,model_name,'model.ckpt-%d' % (step)))
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        for epoch in range(args.epochs):
            if epoch%args.test_freq==0:
                detector_val_logits,detector_val_labels,classifier_MSR_val = [],[],[]
                for batch_num in range(val_batches_per_epoch):
                    cur_logits,cur_labels,cur_softmax = sess.run([detector_logit,detector_labels,classifier_softmax],feed_dict={data2use:'val',lr:cur_lr,keep_prob:1,bn_learning:False})
                    detector_val_logits.append(cur_logits)
                    detector_val_labels.append(cur_labels)
                    classifier_MSR_val.append(np.max(cur_softmax,axis=1))
                val_AUROC_CFI,val_AUROC_MSR = example_utils.ProcessValidationData(detector_val_logits, detector_val_labels, classifier_MSR_val,figures_folder=args.figures_folder)
                detector_train_logits,detector_train_labels,classifier_MSR_train = [],[],[]
                for batch_num in range(train_batches_per_epoch):
                    cur_logits, cur_labels, cur_softmax = sess.run([detector_logit, detector_labels, classifier_softmax],feed_dict={data2use: 'stats', lr: cur_lr, keep_prob: 1, bn_learning: False})
                    detector_train_logits.append(cur_logits)
                    detector_train_labels.append(cur_labels)
                    classifier_MSR_train.append(np.max(cur_softmax, axis=1))
                train_AUROC_CFI,train_AUROC_MSR = example_utils.ProcessValidationData(detector_train_logits, detector_train_labels, classifier_MSR_train)
                print('Epoch %d. AUROC on e(training/validation) sets: (%.3f/%.3f). AUROC using MSR: (%.3f/%.3f)'%(epoch,train_AUROC_CFI,val_AUROC_CFI,train_AUROC_MSR,val_AUROC_MSR))
            train_loss = 0
            for batch_num in range(train_batches_per_epoch):
                step, cur_loss,_ = sess.run([global_step_detector,square_loss,train_op],feed_dict={data2use:'train',lr:cur_lr,keep_prob:keep_prob_,bn_learning:True})
                train_loss += cur_loss/train_batches_per_epoch
            if lowest_train_loss is None or train_loss<lowest_train_loss:
                # print('Minimum train loss drop from %.3f to %.3f'%(lowest_train_loss,train_loss))
                lowest_train_loss = train_loss
                latest_loss_drop = epoch
            elif epoch-latest_loss_drop>args.lr_decrease_epochs:
                latest_loss_drop = epoch
                cur_lr *= LEARNING_RATE_DECAY_FACTOR
                if cur_lr<MIN_LR:
                    print('Breaking training because learning rate is below %.3e'%(MIN_LR))
                    break
                print('Dropping learning rate to %.2e after %d epochs without loss drop. Lowest training loss so far was %.4f.'%(cur_lr,args.lr_decrease_epochs,lowest_train_loss))
        saver_D.save(sess, os.path.join(args.detector_checkpoint,model_name,'model.ckpt-%d' % (step)))
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)