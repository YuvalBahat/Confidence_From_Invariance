# Confidence From Invariance
This is the official TensorFlow implementation of the error and novelty detection method described in the arXiv paper [Confidence from Invariance to Image Transformations](https://arxiv.org/abs/1804.00657) by Yuval Bahat and Gregory Shakhnarovich.

The core idea behind this method is to base an image classification confidence measure on the variations in the classifier's outputs corresponding to different natural transformations of the image at hand. This is done by training a simple multi-layer-perceptron on a given classifier's outputs corresponding to the original image and its transformed versions. Please see the paper for more details.
![sketch](system_overview.png)

If you find our work useful in your research or publication, please cite it:

```
@article{bahat2018confidence,
  title={Confidence from Invariance to Image Transformations},
  author={Bahat, Yuval and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1804.00657},
  year={2018}
}
```
----------
# Usage:

The core transformations TensorFlow operators are implemented in Transformations.py. Use 
