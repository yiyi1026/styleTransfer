# import numpy as np
# from keras.models import Sequential
# from keras.layers import InputLayer, Input
# # from keras.layers import Reshape, MaxPooling2D
# from keras.layers import Conv2D, Dense, Flatten
# # # , Dropout, Activation

# # model = Sequential()
# # # import vgg16
# # from vgg16 import VGG16
# model = VGG19()

# # test session
# filename = "images/starry_night.jpg"

from __future__ import print_function
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b # ?
import time # ?

import argparse

parser = argparse.ArgumentParser(description="Neural style transfer")
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# dimensions of the generated picture.
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# util function to open, resize and format pictures into appropriate tensors

def preprocess_image(image_path):
  # load image and save as JpegImageFile
  img = load_img(image_path, target_size = (img_nrows, img_ncols))

  # transfer JpegImageFile into ndarray, which is a multidimensional, fixed size array object
  img = img_to_array(img)

  # reshapl 3-dimension img into 4-dimension img (by add a dimension before other original dims)
  img = np.expand_dims(img, axis=0)
  # preprocess image so as to be compatible
  np_img = preprocess_input(img)
  return np_img

# util function to convert a tensor into a valid image

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
        # x = K.permute_dimensions(x, (1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# this will contain our generated image
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)
print(input_tensor.shape) # [3, 400, 533, 3]
# build the VGG16 network with our 3 images as input
# the model will be loaded with pre-trained ImageNet weights
model = VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image


def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent


def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocess_image(base_image_path)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    # x is ndarray
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))

# print(model.summary())

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, 224, 224, 3)       0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
# _________________________________________________________________
# flatten (Flatten)            (None, 25088)             0         
# _________________________________________________________________
# fc1 (Dense)                  (None, 4096)              102764544 
# _________________________________________________________________
# fc2 (Dense)                  (None, 4096)              16781312  
# _________________________________________________________________
# predictions (Dense)          (None, 1000)              4097000   
# =================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0

# include_top (True): Whether or not to include the output layers for the model. You don’t need these if you are fitting the model on your own problem.
# weights (‘imagenet‘): What weights to load. You can specify None to not load pre-trained weights if you are interested in training the model yourself from scratch.
# input_tensor (None): A new input layer if you intend to fit the model on new data of a different size.
# input_shape (None): The size of images that the model is expected to take if you change the input layer.
# pooling (None): The type of pooling to use when you are training a new set of output layers.
# classes (1000): The number of classes (e.g. size of output vector) for the model.
