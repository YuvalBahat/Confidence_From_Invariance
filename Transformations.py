import numpy as np
from scipy.misc import imread, imresize
# import cv2
import os
import warnings
import ED_utils
import tensorflow as tf


OPTIONAL_LOGITS_PERTURBATIONS = {0:'horFlip',1:'plusBright',2:'minusBright',3:'increaseContrast3',4:'decreaseContrast',5:'zoom_in',
    6:'horTrans',7:'increaseContrast1',8:'increaseContrast2',9:'BW',11:'masking05',12:'masking10',13:'masking15',\
    21:'gamma07',22:'gamma13',23:'gamma17',24:'gamma05',25:'gamma09',26:'gamma9.5',27:'gamma08',28:'gamma7.5',29:'gamma8.5',31:'gamma10.5',
    32:'gamma11',33:'gamma11.5',41:'noise1',42:'noise3',51:'blur3',52:'blur3',53:'blur7'}

PLUS_BRUGHTNESS_AUGMENTATION = 25
TIMES_CONTRAST_AUGMENTATION = 0.3
ZOOM_IN_FACTOR = 1.05
PIX2TRANSLATE = 10
MAX_PIXEL_VALUE = 255
VIEW_MANIPULATIONS = False

#Utility functions:
def BatchImresize(images, size):
    batch_shape = images.shape[:-3]
    cropped_shape = images.shape[-3:]
    images = np.reshape(images, [np.prod(images.shape[:-3]), -1])
    return np.array(
        list(map(lambda cur_im: imresize(np.reshape(cur_im, cropped_shape), size=size), list(images)))).reshape(list(batch_shape) + list(size) + [3])

#This is the Transformer class (I call the transformations manipulations):
class Transformer():
    def __init__(self, transformations, batch_operation=True):
        # manipulations_string - A string of desired perturbation numbers (using the OPTIONAL_LOGITS_PERTURBATIONS legend), separated by '_'
        # image_size - Size of input images
        # One of the following two arguments has to be set:
        # num_output_images - If the perturbed images were created using a diffrent Manipulator objet, set the number of images
        #   (including their perturbations) in order to perform some checks.
        # num_input_images - In case the number of images (excluding their perturbations) is known, set it here.
        self.batch_operation = batch_operation
        self.transformations = transformations
        # assert self.manipulations_string == transformations,\
        #     'Manipulation string %s is invalid. Please change to %s and re-run.'%(transformations, self.manipulations_string)
        self.num_transformations = len(self.transformations)
        self.per_image_copies = len(self.transformations) + 1
        self.transformation_param = np.nan*np.zeros([self.num_transformations])
        for transformation_num,cur_transformation in enumerate(self.transformations):
            is_digit = [character.isdigit() for character in self.transformations[transformation_num]]
            if any(is_digit):
                first_digit_index = is_digit.index(True)
                self.transformation_param[transformation_num] = float(cur_transformation[first_digit_index:])
                self.transformations[transformation_num] = cur_transformation[:first_digit_index]
    def TransformImages_TF_OP(self,images,labels):
        # Creating the perturbed images.
        # input_image - A single image tensor, in the format [Height,Width,# Channels].
        # In case the images are in a different format, use data_file_config
        # input_label - The corresponding label
        # batch_size - The number of images in the input tensor to this function
        # num_of_KLD_only_per_batch - The number of unlabeled images in input tensor, used only for KL divergence loss. These are assumed to be put after the labeled images in each batch.
        # if set to -1, this is the second call of this function, when the first call was with num_of_KLD_only_per_batch>0, and it yields output corresponding only to the labeled images (for evaluation purposes).
        # Returns:
        # images2use,labels2use
        # if self.TF_batch_size!=1:
        # images_batch = input_image
        # labels_batch = input_label
        # else:
        #     assert self.TF_num_of_KLD_only_per_batch==0,'Currently not supporting batch_size=1 and images with exclusive KLD loss'
        # output_images,output_labels = [],[]
        # for image_num in range(self.TF_batch_size):
        # if self.TF_batch_size!=1:
        #     input_image = tf.reshape(tf.slice(images_batch,begin=[image_num,0,0,0],size=[1,-1,-1,-1]),images_batch.get_shape()[1:])
        #     input_label = tf.reshape(tf.slice(labels_batch,begin=[image_num],size=[1]),[])
        tf.Assert(tf.logical_and(tf.reduce_max(images)>=int(MAX_PIXEL_VALUE/3),tf.reduce_min(images)>=0),[tf.reduce_min(images),tf.reduce_max(images)])
        assert (len(images.get_shape())==3 and not self.batch_operation) or (len(images.get_shape())==4 and self.batch_operation),'Incorrect shape of images input'
        if not self.batch_operation:
            images = tf.expand_dims(images,axis=0)
        image_shape = images.get_shape().as_list()[1:]
        # image_shape = image_shape[1:] if len(image_shape)<3 else image_shape[-3:]
        non_modified_images = tf.cast(images,tf.float32)
        if any([('Contrast' in augm) for augm in self.transformations]):
            image_mean = tf.reduce_mean(images,axis=(1,2),keep_dims=True)
        images2use = tf.expand_dims(images,axis=1)
        for ind,cur_transformation in enumerate(self.transformations):
            if 'increaseContrast' in cur_transformation:
                modified_image = tf.maximum(0.0,tf.minimum((non_modified_images-image_mean)*(1+0.1*self.transformation_param[ind])+image_mean,MAX_PIXEL_VALUE))
            if 'horFlip' in cur_transformation:
                # modified_image = tf.image.flip_left_right(non_modified_images)
                modified_image = tf.map_fn(lambda image: tf.image.flip_left_right(image), non_modified_images)
            if 'blur' in cur_transformation:
                blur_pixels = int(self.transformation_param[ind])
                assert blur_pixels>=2,'Blurring the image with blur kernel of size %d makes no difference'%(blur_pixels)
                pre_blur_images = tf.pad(non_modified_images,paddings=((0,0),(0,0),(int((blur_pixels-1)/2),int((blur_pixels-1)/2)),(0,0)),mode='SYMMETRIC')
                modified_image = tf.zeros_like(non_modified_images)
                for pixel_num in range(blur_pixels):
                    modified_image = tf.add(modified_image,tf.slice(pre_blur_images/blur_pixels,begin=[0,0,pixel_num,0],size=[-1,-1,tf.shape(non_modified_images)[1],-1]))
            if 'BW' in cur_transformation:
                modified_image = tf.tile(tf.reduce_sum(tf.multiply(non_modified_images,tf.reshape(tf.constant([0.299,0.587,0.114]),[1,1,1,3])),axis=3,keep_dims=True),multiples=[1,1,1,3])
            if 'gamma' in cur_transformation:
                modified_image = tf.clip_by_value(non_modified_images,clip_value_min=0,clip_value_max=MAX_PIXEL_VALUE)
                # tf.Assert(tf.reduce_all(tf.greater_equal(non_modified_images,0)),[tf.reduce_min(non_modified_images)])
                modified_image = tf.pow(modified_image/MAX_PIXEL_VALUE,0.1*self.transformation_param[ind])*MAX_PIXEL_VALUE
            images2use = tf.concat((images2use,tf.expand_dims(tf.cast(modified_image,images.dtype),axis=1)),axis=1)
        output_images =  tf.reshape(images2use,[-1]+image_shape)
        # if not self.batch_operation:
        #     output_images = [tf.reshape(im,image_shape) for im in tf.split(output_images,self.per_image_copies,axis=0)]
        #     output_labels = [labels for i in range(self.per_image_copies)]
        # else:
        output_labels = tf.reshape(tf.tile(tf.expand_dims(labels, axis=1), multiples=[1, self.per_image_copies]),[-1])

        return output_images,output_labels
    def Process_NonLogits_TF_OP(self,input_tensor):
        # After running a classifier on the perturbed images, all outputs (but the logits) repeat themselves per_image_copies number
            #  of times. In all outputs but the logits, we are only interested in the output for the original images. This function gets such output
            #   and returns the relevant portion of it.
        input_tensor_shape = input_tensor.get_shape().as_list()
        # print(input_tensor.get_shape())
        if len(input_tensor_shape)>1:
            input_tensor = tf.reshape(input_tensor,[-1,self.per_image_copies]+input_tensor_shape[1:])
            # print(input_tensor.get_shape())
            tensor2return = tf.reshape(tf.slice(input_tensor,begin=[0,0]+list(np.zeros([len(input_tensor_shape)-1]).astype(np.int32)),size=[-1,1]+list(-1*np.ones([len(input_tensor_shape)-1]).astype(np.int32))),
                [-1]+input_tensor_shape[1:])
            if self.TF_num_of_KLD_only_per_batch>0:
                tensor2return = tf.slice(tensor2return,begin=[0]+list(np.zeros([len(input_tensor_shape)-1]).astype(np.int32)),size=[self.TF_batch_size]+list(-1*np.ones([len(input_tensor_shape)-1]).astype(np.int32)))
            # print(tensor2return.get_shape())
        else:
            input_tensor = tf.reshape(input_tensor,[-1,self.per_image_copies])
            # print(input_tensor.get_shape())
            tensor2return = tf.reshape(tf.slice(input_tensor,begin=[0,0]+list(np.zeros([len(input_tensor_shape)-1]).astype(np.int32)),size=[-1,1]+list(-1*np.ones([len(input_tensor_shape)-1]).astype(np.int32))),[-1])
            # if self.TF_num_of_KLD_only_per_batch>0:
            #     tensor2return = tf.slice(tensor2return,begin=[0],size=[self.TF_batch_size])
        return tf.stop_gradient(tensor2return)
    def Process_Logits_TF_OP(self,input_logits,reorder_logits=True,num_logits_per_transformation=-1):
        # For the logits output of a classifier, this function convets it to feature vector for our detector. if GT_labels for the images
        # are given, the function returns the detector labels as well (whether the image was correctly or incorrectly classified).
        #top5 - Concerns the detector labels - If true, the detector label is True if the correct label is not among the highest 5 logits.
            # The False option calls for using a per-predicted-label detector.
        # Returns:
            # input_logits - the logits of the orignal images (similar to the output of Process_NonLogits
            # features_vect -the feature vectors comprised by concatenating the logits of the original images and those
                # of the different perturbations
            # detector_label - The detector labels, when GT_labels are given.
        input_logits_shape = input_logits.get_shape().as_list()
        assert len(input_logits_shape)==2,'Unrecognized logits shape'
        assert not (num_logits_per_transformation>0 and not reorder_logits),'Cannot keep k logits per transformation without sorting them'
        assert not num_logits_per_transformation>input_logits_shape[1],'Cannot keep more logits (%d) than there are originally (%d)'%(num_logits_per_transformation,input_logits_shape[1])
        # assert not (KLD_loss_output and TVD_loss_output),'Should choose either flags'
        input_logits = tf.reshape(input_logits,[-1,self.per_image_copies,input_logits_shape[1]])
        logits_of_original = tf.reshape(tf.slice(input_logits,begin=[0,0,0],size=[-1,1,-1]),[-1,input_logits_shape[-1]])
        if reorder_logits:
            org_logits_shape = logits_of_original.get_shape().as_list()
            _,descending_order = tf.nn.top_k(logits_of_original,k=10)
            descending_order = tf.tile(tf.reshape(descending_order,shape=[-1,1,org_logits_shape[1],1]),[1,self.per_image_copies,1,1])
            image_indices = tf.tile(tf.reshape(tf.range(org_logits_shape[0]),[org_logits_shape[0],1,1,1]),multiples=[1,self.per_image_copies,org_logits_shape[1],1])
            permutation_indices = tf.tile(tf.reshape(tf.range(self.per_image_copies),[1,self.per_image_copies,1,1]),[org_logits_shape[0],1,org_logits_shape[1],1])
            combined_indices = tf.concat([image_indices,permutation_indices,descending_order],axis=3)
            descending_values = tf.gather_nd(params=input_logits,indices=combined_indices)
            if num_logits_per_transformation>0:
                descending_values = tf.slice(descending_values,begin=[0,0,0],size=[-1,-1,num_logits_per_transformation])
            features_vect = tf.reshape(descending_values,[int(input_logits_shape[0]/self.per_image_copies),-1])
        else:
            features_vect = tf.reshape(input_logits,[-1, self.per_image_copies * input_logits_shape[-1]])
        # input_logits = tf.reshape(tf.slice(input_logits,begin=[0,0,0],size=[-1,1,-1]),[-1,input_logits_shape[-1]])
        return tf.stop_gradient(logits_of_original),tf.stop_gradient(features_vect)