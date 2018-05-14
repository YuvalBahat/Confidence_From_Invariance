import cifar10.cifar10 as cifar10
import tensorflow as tf

DATASET_FOLDER = '/home/ybahat/data/Databases/cifar10/bin'
class example_classifier:
    def __init__(self,checkpoint_dir,images,labels):
        # with tf.Graph().as_default() as g:
        # self.images_train, self.labels_train = cifar10.inputs(eval_data=False, inner_data_dir=DATASET_FOLDER)
        #
        # def TrainData():
        #     return self.images_train, self.labels_train
        #
        # self.images_val, self.labels_val = cifar10.inputs(eval_data=True, inner_data_dir=DATASET_FOLDER)
        #
        # def ValData():
        #     return self.images_val, self.labels_val
        #
        # # self.val_data = tf.placeholder(dtype=tf.bool)
        # self.images,self.labels = tf.cond(val_data,true_fn=ValData,false_fn=TrainData)
        self.classifier = cifar10.inference(images)
        self.logits = self.classifier.inference_logits()
        self.correctness = tf.nn.in_top_k(self.logits,labels, 1)

        # Restore the moving average version of the learned variables for eval.
        # variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(variables_to_restore)
        # with tf.Session() as sess:
        #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        #     if ckpt and ckpt.model_checkpoint_path:
        #         # Restores from checkpoint
        #         saver.restore(sess, ckpt.model_checkpoint_path)
        #     else:
        #       print('No checkpoint file found')
        #       return
    def ClassifyBatch(self):
        # with session as sess:
        return self.logits,self.correctness
