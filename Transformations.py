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
def ModificationName(modification_num):
    if isinstance (modification_num,list):
        result = [OPTIONAL_LOGITS_PERTURBATIONS[i] for i in modification_num]
        return [(element+'3' if element=='increaseContrast' else element) for element in result]
    else:
        result = OPTIONAL_LOGITS_PERTURBATIONS[modification_num]
        return result+'3' if result=='increaseContrast' else result
def Str2Modifs(augmentation_strings):
    augmentation_strings = augmentation_strings.split('_')
    sorted_augms = sorted(augmentation_strings)
    resorted_strings = '_'.join(sorted_augms)
    augm2couple = []
    for augmentation in sorted_augms:
        augm2couple.append(OPTIONAL_LOGITS_PERTURBATIONS[int(augmentation)])
    return augm2couple,resorted_strings
def BatchImresize(images, size):
    batch_shape = images.shape[:-3]
    cropped_shape = images.shape[-3:]
    images = np.reshape(images, [np.prod(images.shape[:-3]), -1])
    return np.array(
        list(map(lambda cur_im: imresize(np.reshape(cur_im, cropped_shape), size=size), list(images)))).reshape(list(batch_shape) + list(size) + [3])

#This is the Transformer class (I call the transformations manipulations):
class Transformer():
    def __init__(self, transformations, image_size=None, num_output_images=None, num_input_images=None):
        # manipulations_string - A string of desired perturbation numbers (using the OPTIONAL_LOGITS_PERTURBATIONS legend), separated by '_'
        # image_size - Size of input images
        # One of the following two arguments has to be set:
        # num_output_images - If the perturbed images were created using a diffrent Manipulator objet, set the number of images
        #   (including their perturbations) in order to perform some checks.
        # num_input_images - In case the number of images (excluding their perturbations) is known, set it here.
        self.transformations = transformations
        # assert self.manipulations_string == transformations,\
        #     'Manipulation string %s is invalid. Please change to %s and re-run.'%(transformations, self.manipulations_string)
        self.num_transformations = len(self.transformations)
        self.per_image_copies = len(self.transformations) + 1
        self.transformation_param = np.nan*np.zeros([self.num_transformations])
        if image_size is not None:
            self.image_size = image_size
        if num_output_images is not None:
            self.num_of_images = int(num_output_images/self.per_image_copies)
            self.num_of_output_images = num_output_images
        elif num_input_images is not None:
            self.num_of_images = num_input_images
            self.num_of_output_images = num_input_images*self.per_image_copies
        else:
            self.num_of_images = None
            self.num_of_output_images = None
        for transformation_num,cur_transformation in enumerate(self.transformations):
            is_digit = [character.isdigit() for character in self.transformations[transformation_num]]
            if any(is_digit):
                first_digit_index = is_digit.index(True)
                self.transformation_param[transformation_num] = float(cur_transformation[first_digit_index:])
                self.transformations[transformation_num] = cur_transformation[:first_digit_index]
    def TransformImages(self,input_images,input_labels,data_file_config=None):
        # Creating the perturbed images.
        # input_images - The images of the dataset, in the format [num_input_images,Height,Width,# Channels].
        # In case the images are in a different format, use data_file_config
        # input_labels - The corresponding labels
        # Returns:
        # images2use,labels2use

        if np.max(input_images)<MAX_PIXEL_VALUE/3 or np.min(input_images)<0:
            warnings.warn('Assuming pixels range is [0,%d], but the actual range is [%.2f,%.2f]'%(MAX_PIXEL_VALUE,np.min(input_images),np.max(input_images)))
        final_shape = input_images.shape[1:]
        if any(np.not_equal(final_shape,self.image_size)):
            if data_file_config=='matlab':
                input_images = np.transpose(input_images.reshape([-1]+list(self.image_size[::-1])),(0,2,3,1))
            elif data_file_config=='python':
                input_images = input_images.reshape([-1]+list(self.image_size))
            else:
                raise Exception('Unrecognized data file configuration %s'%(data_file_config))
        if self.num_of_images is None:
            self.num_of_images = input_images.shape[0]
            self.num_of_output_images = input_images.shape[0]*self.per_image_copies
        non_modified_images = np.expand_dims(input_images,axis=1)
        if any([('Contrast' in augm) for augm in self.transformations]):
            images_mean = np.mean(non_modified_images,axis=(2,3),keepdims=True)
        images2use = np.expand_dims(input_images,axis=1)
        if VIEW_MANIPULATIONS:
            import matplotlib.pyplot as plt
            plt.imsave('before.png',non_modified_images[0,0,...].astype(np.uint8))
        for ind,cur_transformation in enumerate(self.transformations):
            modified_images = non_modified_images
            if 'increaseContrast' in cur_transformation:
                modified_images = np.maximum(0,np.minimum((modified_images-images_mean)*(1+0.1*self.transformation_param[ind])+images_mean,MAX_PIXEL_VALUE))
            if 'decreaseContrast' in cur_transformation:
                modified_images = np.maximum(0,np.minimum((modified_images-images_mean)*(1-0.1*self.transformation_param[ind])+images_mean,0),MAX_PIXEL_VALUE)
            if 'zoom_in' in cur_transformation:
                pixels2omit = int(non_modified_images.shape[2]*(1-1/ZOOM_IN_FACTOR))
                modified_images = modified_images[:,:,int(np.floor(pixels2omit/2)):-int(np.ceil(pixels2omit/2)),int(np.floor(pixels2omit/2)):-int(np.ceil(pixels2omit/2)),:]
                modified_images = BatchImresize(modified_images,size=non_modified_images.shape[2:4])
            if 'horFlip' in cur_transformation:
                modified_images = modified_images[:,:,:,::-1,:]
            if 'horTrans' in cur_transformation:
                modified_images = BatchImresize(modified_images[:,:,:,PIX2TRANSLATE:,:],size=non_modified_images.shape[2:4])
            if 'plusBright' in cur_transformation:
                modified_images = np.minimum(modified_images+PLUS_BRUGHTNESS_AUGMENTATION,MAX_PIXEL_VALUE)
            if 'minusBright' in cur_transformation:
                modified_images = np.maximum(modified_images-PLUS_BRUGHTNESS_AUGMENTATION,0)
            if 'masking' in cur_transformation:
                cur_mask = np.reshape(self.uniform_noise_image>0.01*self.transformation_param[ind],[1,1]+self.image_size[:2]+[1]).astype(modified_images.dtype)
                print('Masking %.3f of pixels'%(1-np.mean(cur_mask)))
                modified_images = cur_mask*modified_images+np.array(self.VGG16_channels_mean).reshape([1,1,1,1,3])*np.ones(self.image_size).reshape([1,1]+self.image_size)*(1-cur_mask)
            if 'blur' in cur_transformation:
                # blur_pixels = int(1+2*np.round(0.01*self.manipulation_param[ind]*modified_images.shape[3]/2))
                blur_pixels = int(self.transformation_param[ind])
                assert blur_pixels>=2,'Blurring the image with blur kernel of size %d makes no difference'%(blur_pixels)
                pre_blur_images = np.pad(modified_images,pad_width=((0,0),(0,0),(0,0),(int((blur_pixels-1)/2),int((blur_pixels-1)/2)),(0,0)),mode='edge')
                modified_images = np.zeros_like(modified_images).astype(np.float32)
                for pixel_num in range(blur_pixels):
                    modified_images += pre_blur_images[:,:,:,pixel_num:pixel_num+self.image_size[1],:]/blur_pixels
            if 'BW' in cur_transformation:
                modified_images = np.repeat(np.sum(modified_images*np.reshape([0.299,0.587,0.114],[1,1,1,1,3]),axis=4,keepdims=True),3,axis=4)
            if 'gamma' in cur_transformation:
                assert np.all(modified_images>=0),'Trying to perform gamma transformation on negative pixel values'
                modified_images = np.power(modified_images.astype(np.float32)/MAX_PIXEL_VALUE,0.1*self.transformation_param[ind])*MAX_PIXEL_VALUE
            if 'noise' in cur_transformation:
                cur_noise = 0.01*self.transformation_param[ind]*self.normal_noise_image.reshape([1,1]+list(self.normal_noise_image.shape))
                # print(np.min(cur_noise),np.max(cur_noise))
                # plt.imsave('before.png',modified_images[0,0,...].astype(np.uint8))
                modified_images = np.maximum(0,np.minimum(MAX_PIXEL_VALUE,(modified_images.astype(np.float32)/MAX_PIXEL_VALUE+cur_noise)*MAX_PIXEL_VALUE))
                # plt.imsave('after.png',modified_images[0,0,...].astype(np.uint8))
                # raise Exception
            images2use = np.concatenate((images2use,modified_images),axis=1)
        if VIEW_MANIPULATIONS:
            plt.imsave('after.png',modified_images[0,0,...].astype(np.uint8));  raise Exception
        if data_file_config=='matlab':
            images2use = np.reshape(np.transpose(images2use,(0,1,4,2,3)), [self.num_of_output_images,-1])
        # elif any(np.not_equal(final_shape,self.image_size)):
        #     images2use = np.reshape(images2use, [self.num_of_output_images] + list(final_shape))
        else:
            images2use = np.reshape(images2use,[self.num_of_output_images]+list(images2use.shape[2:]))
        label_output_shape = list(input_labels.shape)
        label_output_shape[0]*=self.per_image_copies
        labels2use = np.tile(np.expand_dims(np.expand_dims(input_labels,axis=1),axis=-1),[1,self.per_image_copies,1]).reshape(label_output_shape)
        return images2use,labels2use
    def TransformImages_TF_OP(self,images,labels,data_file_config=None):
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
                tf.Assert(tf.reduce_all(tf.greater_equal(non_modified_images,0)),[tf.reduce_min(non_modified_images)])
                modified_image = tf.pow(non_modified_images/MAX_PIXEL_VALUE,0.1*self.transformation_param[ind])*MAX_PIXEL_VALUE
            images2use = tf.concat((images2use,tf.expand_dims(tf.cast(modified_image,images.dtype),axis=1)),axis=1)
        output_images =  tf.reshape(images2use,[-1]+image_shape)
        output_labels = tf.reshape(tf.tile(tf.expand_dims(labels,axis=1),multiples=[1,self.per_image_copies]),[-1])
        return output_images,output_labels
    def Process_NonLogits(self,input_array):
        # After running a classifier on the perturbed images, all outputs (but the logits) repeat themselves per_image_copies number
            #  of times. In all outputs but the logits, we are only interested in the output for the original images. This function gets such output
            #   and returns the relevant portion of it.
        assert np.mod(input_array.shape[0],self.per_image_copies)==0,'Input size is not an integer multiplication of number of transformations'
        assert input_array.shape[0]==self.num_of_output_images,'Expected %d rows but got %d'%(self.num_of_output_images,input_array.shape[0])
        return np.reshape(input_array,[self.num_of_images,self.per_image_copies]+list(input_array.shape[1:]))[:,0,...].reshape([self.num_of_images]+list(input_array.shape[1:]))
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
    def Process_Logits(self,input_logits,GT_labels=None,file_name=None,sort_logits=False,top5=False):
        # For the logits output of a classifier, this function convets it to feature vector for our detector. if GT_labels for the images
        # are given, the function returns the detector labels as well (whether the image was correctly or incorrectly classified).
        # sort_logits - if True, logits in the feature vector are sorted in decreasing order of the logits of the original images,
            # for all perturbations. This makes the yielded feature vector look similar for images that correspond to different labels,
            # at the cost of loosing some information. If False, the original order of logits is maintained. This keeps the information
            # about order of logits, at the cost of having very different feature vectors for images corresponding to different labels.
        #top5 - Concerns the detector labels - If true, the detector label is True if the correct label is not among the highest 5 logits.
            # The False option calls for using a per-predicted-label detector.
        # Returns:
            # input_logits - the logits of the orignal images (similar to the output of Process_NonLogits
            # features_vect -the feature vectors comprised by concatenating the logits of the original images and those
                # of the different perturbations
            # detector_label - The detector labels, when GT_labels are given.
        assert np.mod(input_logits.shape[0],self.per_image_copies)==0,'Input size is not an integer multiplication of number of transformations'
        assert input_logits.shape[0]==self.num_of_output_images,'Expected %d rows of logits but got %d'%(self.num_of_output_images,input_logits.shape[0])
        assert np.all(np.logical_not(np.isnan(input_logits))),'%.3e of logits is nan'%(np.mean(np.isnan(input_logits)))
        input_logits = input_logits.reshape([self.num_of_images,self.per_image_copies,input_logits.shape[-1]])
        if sort_logits:
            logits_reordering = np.argsort(input_logits[:, 0, :], axis=1)[:, ::-1]
            features_vect = input_logits[np.arange(input_logits.shape[0]).reshape([-1, 1, 1]), np.arange(self.per_image_copies).reshape([1, -1, 1]),
                                         np.expand_dims(logits_reordering, axis=1)]
            features_vect = np.reshape(features_vect,
                                       [features_vect.shape[0], self.per_image_copies * input_logits.shape[-1]])
        else:
            features_vect = np.reshape(input_logits,[input_logits.shape[0], self.per_image_copies * input_logits.shape[-1]])
        input_logits = input_logits[:,0,:]
        if GT_labels is None:
            return input_logits,features_vect
        else:
            if top5:
                if GT_labels.size==GT_labels.shape[0]:#GT_labels are indecis of the correct class
                    detector_label = np.all(np.not_equal(np.argsort(input_logits,axis=1)[:,-5:],GT_labels.reshape([-1,1])),axis=1)
                else:#GT labels are in the form of 1-hot vectors
                    raise Exception('Not implemented')
            else:
                if GT_labels.size==GT_labels.shape[0]:#GT_labels are indecis of the correct class
                    detector_label = np.not_equal(np.argmax(input_logits,axis=1),GT_labels.reshape([-1]))
                else:#GT labels are in the form of 1-hot vectors
                    detector_label = np.not_equal(np.argmax(input_logits,axis=1),np.argmax(GT_labels,axis=1))
            # if file_name is not None:
            #     np.savez(file_name+'_'+self.manipulations_string+'.npz',features_vect=features_vect,detector_label=detector_label)
            return input_logits,features_vect,detector_label
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
def KL_divergence(logits_of_transformations,num_transformations):
    # Given feature vectors of logits (the second output of Process_Logits), returns the KL-divergence of each of the perturbation
        #  for each of the images.
    import scipy.stats as stats
    num_logits_per_copy = int(logits_of_transformations.shape[1]/(num_transformations+1))
    KL_divergences = np.nan*np.zeros([logits_of_transformations.shape[0],num_transformations])
    cur_p_set = ED_utils.soft_max(logits_of_transformations[:, :num_logits_per_copy])
    for transformation_num in range(num_transformations):
        cur_q_set = ED_utils.soft_max(logits_of_transformations[:, (transformation_num+1) * num_logits_per_copy:(transformation_num+2) * num_logits_per_copy])
        cur_q_set[np.logical_and(cur_q_set == 0, cur_p_set != 0)] = np.nextafter(np.array(0).astype(cur_q_set.dtype),
                                                                                 np.array(1).astype(cur_q_set.dtype))
        cur_p_set[np.logical_and(cur_p_set == 0, cur_q_set != 0)] = np.nextafter(np.array(0).astype(cur_p_set.dtype),
                                                                                 np.array(1).astype(cur_p_set.dtype))
        for sample_num in range(logits_of_transformations.shape[0]):
            KL_divergences[sample_num,transformation_num] = stats.entropy(pk=cur_p_set[sample_num, :], qk=cur_q_set[sample_num, :])
    return KL_divergences
