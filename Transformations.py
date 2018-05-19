import numpy as np
import tensorflow as tf

# Implemented transformations:
#   'horFlip':              Horizontal image flip
#   'increaseContrast<x>':  Increase image contrast by 0.1*x (e.g. 'increaseContrast3': Increase contrast by 0.3)
#   'BW':                   Convert image to gray-scale
#   'gamma<x>':             Gamma transformation. Each pixel value is raised to the power of 0.1*x, after being normalized to the range [0,1]
#   'blur<x>':              Horizontally blur the image with a magnitude of x pixels.

class Transformer():
    def __init__(self, transformations, batch_operation=True,max_pixel_value=255):
        # Inputs:
        #   transformations:    List of strings (out of the optional tranformations listed above) defining the transformations to be applied on all images.
        #                       Each string may optionally contain a parameter, in the form of <transformation_name><parameter> (e.g. gamma8.5)
        #   batch_operation:    Whether the transformer should operate on a single images tensor or a batch of images tensor (default)
        #   max_pixel_value:    Pixels' dynamic range is expected to be in [0,max_pixel_value]
        # Output:
        #   A transformer object

        self.batch_operation = batch_operation
        self.max_pixel_value = max_pixel_value
        self.transformations = transformations
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
        # Creating the transformed images and labels.
        # Inputs:
        #   images: A single image (HxWxC) or a batch of images (NxHxWxC) tensor (depending on batch_operation)
        #   labels: A 1-D tensor of corresponding image labels
        # Outputs:
        #   output_images:  A batch of images and their transformed versions. Each image is followed by its transformed versions, then by the next image (if batch_operation).
        #   output_labels:  A 1-D batch of corresponding labels
        assert (len(images.get_shape())==3 and not self.batch_operation) or (len(images.get_shape())==4 and self.batch_operation),'Incorrect shape of images input'
        if not self.batch_operation:
            images = tf.expand_dims(images,axis=0)
        image_shape = images.get_shape().as_list()[1:]
        non_modified_images = tf.cast(images,tf.float32)
        if any([('Contrast' in augm) for augm in self.transformations]):
            image_mean = tf.reduce_mean(images,axis=(1,2),keep_dims=True)
        images2use = tf.expand_dims(images,axis=1)
        for ind,cur_transformation in enumerate(self.transformations):
            if 'increaseContrast' in cur_transformation:
                modified_image = tf.maximum(0.0,tf.minimum((non_modified_images-image_mean)*(1+0.1*self.transformation_param[ind])+image_mean,self.max_pixel_value))
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
                modified_image = tf.clip_by_value(non_modified_images,clip_value_min=0,clip_value_max=self.max_pixel_value)
                # tf.Assert(tf.reduce_all(tf.greater_equal(non_modified_images,0)),[tf.reduce_min(non_modified_images)])
                modified_image = tf.pow(modified_image/self.max_pixel_value,0.1*self.transformation_param[ind])*self.max_pixel_value
            images2use = tf.concat((images2use,tf.expand_dims(tf.cast(modified_image,images.dtype),axis=1)),axis=1)
        output_images =  tf.reshape(images2use,[-1]+image_shape)
        output_labels = tf.reshape(tf.tile(tf.expand_dims(labels, axis=1), multiples=[1, self.per_image_copies]),[-1])

        return output_images,output_labels

    def Process_NonLogits_TF_OP(self,input_tensor):
        # The outputs of a classifier fed with transformed images will often correspond to its input batch size, which was modified by the transformer.
        # This function disimply scards the outputs that correspond to the transformed images and outputs only those corresponding to the original images.
        # To be used on all output tensors (e.g. "correct classification" tensor) but the logits tensor
        # Input:
        #   A tensor whose first dimension corresponds to the classifier's input batch size
        # Output: (I apply tf.stop_gradient on the output to avoid the unnecessary gradients calculation when training a detector)
        #  The input tensor after filtering out values corresponding to the transformed images (along the first dimension)

        input_tensor_shape = input_tensor.get_shape().as_list()
        if len(input_tensor_shape)>1:
            input_tensor = tf.reshape(input_tensor,[-1,self.per_image_copies]+input_tensor_shape[1:])
            tensor2return = tf.reshape(tf.slice(input_tensor,begin=[0,0]+list(np.zeros([len(input_tensor_shape)-1]).astype(np.int32)),size=[-1,1]+list(-1*np.ones([len(input_tensor_shape)-1]).astype(np.int32))),
                [-1]+input_tensor_shape[1:])
        else:
            input_tensor = tf.reshape(input_tensor,[-1,self.per_image_copies])
            tensor2return = tf.reshape(tf.slice(input_tensor,begin=[0,0]+list(np.zeros([len(input_tensor_shape)-1]).astype(np.int32)),size=[-1,1]+list(-1*np.ones([len(input_tensor_shape)-1]).astype(np.int32))),[-1])
        return tf.stop_gradient(tensor2return)

    def Process_Logits_TF_OP(self,input_logits,reorder_logits=True,num_logits_per_transformation=-1):
        # Converts the logit output of a classifier fed by transformed images into a logits vector corresponding to the original image and a features vector.
        # Inputs:
        #   input_logits:                   The logits output of a classifier of interest, in the shape of NxNUM_CLASSES,
        #                                   where N is the original batch size X (number of transformations+1)
        #   reorder_logits:                 If True (default), the features vector has the logits corresponding to all transformations (including the original
        #                                   non-transformed image) ordered according to a descending order of the logits corresponding to the original image.
        #   num_logits_per_transformation:  (optional) Using only the logits corresponding to the top num_logits_per_transformation logits of the original image.
        #                                   To use this option, pass an integer between 1 and NUM_CLASSES-1
        # Outputs: (I apply tf.stop_gradient on both outputs to avoid the unnecessary gradients calculation when training a detector)
        #   logits_of_original: Logits tensors corresponding to the original, non-transformed, image. The logits corresponding to the transformed versions are removed.
        #   features_vect:      Feature vectors tensor of shape N x (number of transformation+1) x min(NUM_CLASSES,num_logits_per_transformation)
        input_logits_shape = input_logits.get_shape().as_list()
        assert len(input_logits_shape)==2,'Unrecognized logits shape'
        assert not (num_logits_per_transformation>0 and not reorder_logits),'Cannot keep k logits per transformation without reordering them'
        assert not num_logits_per_transformation>input_logits_shape[1],'Cannot keep more logits (%d) than there are originally (%d)'%(num_logits_per_transformation,input_logits_shape[1])
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
        return tf.stop_gradient(logits_of_original),tf.stop_gradient(features_vect)