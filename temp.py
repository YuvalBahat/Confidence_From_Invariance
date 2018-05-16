# import cifar10.example_classifier as example_classifier
import cifar10.cifar10 as cifar10
import tensorflow as tf
import os
from example_utils import SplitCifar10TestSet
import numpy as np
import Transformations

if '/ybahat/PycharmProjects/' in os.getcwd():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Limit to 1 GPU when using an interactive session
else:
    print('Not limiting to 1 GPU')

NUM_OF_SAMPLES = 10000
TRAIN_PORTION = 0.5
DATASET_FOLDER = '/home/ybahat/data/Databases/cifar10/bin'
TRANSFORMATIONS_LIST = ['BW','horFlip','increaseContrast3','gamma8.5','blur3']
BATCH_SIZE = 32
IMAGE_SIZE = [32,32,3]
# with tf.Session() as sess:
validation_split_filename = os.path.join(DATASET_FOLDER,'ValidationSetSplit_%s.npz'%(str(TRAIN_PORTION).replace('.','_')))
if os.path.exists(validation_split_filename):
    detector_train_set_indicator = np.load(validation_split_filename)['detector_train_set_indicator']
else:
    detector_train_set_indicator = np.zeros([NUM_OF_SAMPLES]).astype(np.bool)
    detector_train_set_indicator[np.random.permutation(NUM_OF_SAMPLES)[:int(NUM_OF_SAMPLES*TRAIN_PORTION)]] = True
    np.savez(validation_split_filename,detector_train_set_indicator=detector_train_set_indicator)
SplitCifar10TestSet(train_indicator=detector_train_set_indicator,dataset_folder=DATASET_FOLDER)
val_data = tf.placeholder(dtype=tf.bool)
images_train,labels_train = cifar10.inputs(eval_data=False, inner_data_dir=DATASET_FOLDER,batch_size=BATCH_SIZE)
def TrainData():    return images_train, labels_train
images_val, labels_val = cifar10.inputs(eval_data=True, inner_data_dir=DATASET_FOLDER,batch_size=BATCH_SIZE)
def ValData():  return images_val, labels_val

images, labels = tf.cond(val_data, true_fn=ValData, false_fn=TrainData)
transformer = Transformations.Transformer(transformations=TRANSFORMATIONS_LIST,image_size=IMAGE_SIZE)
images,labels = transformer.TransformImages_TF_OP(images,labels)
classifier = cifar10.inference(images)
logits = classifier.inference_logits()
correctness = tf.nn.in_top_k(logits,labels, 1)
logits = transformer.Process_Logits_TF_OP(logits)[1]
detector_labels = tf.logical_not(transformer.Process_NonLogits_TF_OP(correctness))
variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/home/ybahat/PycharmProjects/CFI/cifar10/checkpoint')
    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    coord = tf.train.Coordinator()
    try:
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
            threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
        cur_logits,cur_correctness = sess.run([logits,detector_labels],feed_dict={val_data:True})
        print(cur_logits.shape,sum(cur_correctness))
    except Exception as e:  # pylint: disable=broad-except
        coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)
