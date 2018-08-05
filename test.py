
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, save_img, img_to_array
import argparse


# In[2]:


def build_parser():
  # arguments preparation
  parser = argparse.ArgumentParser(description="Style Transfer")

  # required arguments
  parser.add_argument('content_img_path', metavar='content', type=str, help='path to content image')
  parser.add_argument('style_img_path', metavar='style', type=str, help='path to style image')
  parser.add_argument('generated_prefix', metavar='gen_prefix', type=str, help='prefix for generated results')

  # optional arguments
  parser.add_argument("--content_weight", type=float, default=0.02, required=False, help="Content Weight")
  parser.add_argument("--style_weight", type=float, default=1, required=False, help="Style Weight")
  parser.add_argument("--tv_weight", type=float, default=1, required=False, help="Total Variation Weight")
  parser.add_argument("--iter", type=int, default=10, required=False, help="Number of Iteration")
  return parser


# In[3]:


# # start
# parser = build_parser()
# # fetch arguments
# args = parser.parse_args()

# fetch required arguments
# content_img_path = args.content_img_path
# style_img_path = args.style_img_path
# generated_prefix = args.generated_prefix

## test for jupyter implementation
content_img_path = 'images/content_small.jpg'
style_img_path = 'images/style_small.jpg'
generated_prefix = ''

# # fetch optional arguments
# content_weight = args.content_weight
# style_weight = args.style_weight
# tv_weight = args.tv_weight
# iteration = args.iter

## test for jupyter implementation
content_weight = 0.02
style_weight = 1
tv_weight = 1
iteration = 10

width, height = load_img(content_img_path).size
# print([width,height])
img_height = 500
img_width = int(width * img_height / height)

# test completed


# In[4]:


def preprocess_img(img_path):
    
  # load image as JpegImageFile
  new_img = load_img(img_path)
  img = load_img(img_path, target_size = (img_height, img_width))

  # convert JpegImageFile into ndarray(a multidimensional, fixed size array object)
  img = img_to_array(img)

  # reshape 3-dimension img into 4-dimension img (by add a dimension upfront)
  img = np.expand_dims(img, axis=0)

  # preprocess image so as to be compatible for keras
  np_img = preprocess_input(img)

  return np_img

def deprocess_img(np_img):
  np_img = np_img.reshape((img_height, img_width, 3))

  # add back testing-mean-pixel (due to default manipulation in openCV from Caffe of Keras)
  np_img[:, :, 0] += 103.939
  np_img[:, :, 1] += 116.779
  np_img[:, :, 2] += 123.68

  # BGR to RGB(due to default manipulation in openCV from Caffe of Keras)
  np_img = np_img[:, :, ::-1]
  img = np.clip(np_img, 0, 255).astype('uint8')
  return img


# In[5]:


# get tensor representations of input images and ouput images in (samples, height, width, channels) shape (1, height, width, channel)
content_img = K.variable(preprocess_img(content_img_path))
style_img = K.variable(preprocess_img(style_img_path))
#### test confirms above

# initial noise image
noise_img = np.random.randint(256, size=(1, img_height, img_width, 3)).astype('float64')
# change???
generated_img =  K.placeholder((1, img_height, img_width, 3))

# print("output")
# print(layer_outputs_dict[model.layers[0].name][1,:,:,:])


# In[6]:


# 4 util loss functions
def content_loss(original, generated):
  return(K.sum(K.square(original - generated)))

# help method for computing style loss
def gram_matrix(input):
  # n-dimensions to 2 dimensions
  temp = K.permute_dimensions(input,(0,1,2))
  # print("temp_value")
  # print(K.get_value(temp))
  features = K.batch_flatten(temp)
  # print("features")
  # print(K.get_value(features))
  # if features is 3*n, return n*n
  return K.dot(K.transpose(features), features)

def style_loss(style, generated):
  style_gram = gram_matrix(style)
  gen_gram = gram_matrix(generated)
  nl = 3
  nl_size = img_height * img_width
  return K.sum(K.square(style_gram-gen_gram))/(4.* (nl ** 2) * (nl_size ** 2))


# In[7]:


style_features_names = ['block1_conv1','block2_conv1','block3_conv1','block4_conv1','block5_conv1']
# generate tensor for 3 images
input_tensor = K.concatenate([content_img, style_img, generated_img], axis=0)
# initialize the VGG19 model
model = VGG19(include_top=False, weights='imagenet', input_tensor = input_tensor)
# store layer.name and layer.output in dictionary
layer_outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# combine the above several loss functions together
def total_loss(content_img, style_imge, generated_img):
  loss = K.variable(value=0.)
  # content_loss
  content_layer_name = 'block5_conv2'
  content_layer_features = layer_outputs_dict[content_layer_name]
  content_features = content_layer_features[0,:,:,:]
  generated_features = content_layer_features[2,:,:,:]
  loss = loss + content_weight * content_loss(content_features, generated_features)

  # style_loss
  active_layers_count = len(style_features_names)
  for layer_name in style_features_names:
    style_layer_features = layer_outputs_dict[layer_name]
    style_features = style_layer_features[1,:,:,:]
    generated_features = style_layer_features[2,:,:,:]
    style_loss_of_this_layer = style_loss(style_features, generated_features)
    loss = loss + style_weight / active_layers_count * style_loss_of_this_layer

  return loss


# In[12]:


l = total_loss(content_img, style_img, generated_img)
# don't know how to get value for this l variable

# reference https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py

# should this line be outside the generated_grad_func or inside?
#grads = K.gradient(l, generated_img)

# generate data point =>loss function
def generated_grad_func(loss, generated_img):
  grads = K.gradients(l, generated_img)
  outputs = [l]
  if isinstance(grads, (list, tuple)):
    outputs += grads
  else:
    outputs.append(grads)

  f_outputs = K.function([generated_img], outputs)
  return f_outputs

f_outputs = generated_grad_func(l, generated_img)

def eval_loss_and_grads(x):
    x = x.reshape((1, img_rows, img_cols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_value = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_value = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grads_value = grad_value
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grads_values = np.copy(self.grads_value)
        self.loss_value = None
        self.grads_value = None
        return grads_values

name = 'name'
i = 3
prefix = name + '_at_iteration_%d.png' % i
# # input_img_data
# save_img(prefix, deprocess_img(noise_img))
save_img(prefix, deprocess_img(noise_img))
# img.show

# a = [[1,2,3,9]]
# c = [[4,5,6,8],[0,1,2,3]]

# inputs = K.placeholder((2, 3))
# print(inputs)
# input_transposed = K.transpose(inputs)
# print(input_transposed)

# t = K.concatenate([a, c], axis=0)
# with tf.Session() as sess:
#   print(t.eval())
  

# b = K.variable([a,c])
# print(b.shape)
# print(K.transpose(0).shape)

# var = K.variable([[1, 2, 3], [4, 5, 6]])
# K.eval(var)

# var_transposed = K.transpose(var)


# In[13]:


def main():
  evaluator = Evaluator()
  input = preprocess_img(content_img_path)
  for i in range(iteration):
    input, min_val, info = fmin_l_bfgs_b(evaluator.loss, input.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print("Current loss ")
  return(i)
main()

