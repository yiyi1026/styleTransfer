import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, save_img, img_to_array
import argparse

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

# fetch arguments
args = parser.parse_args()

# fetch required arguments
content_img_path = args.content_img_path
style_img_path = args.style_img_path
generated_prefix = args.generated_prefix

# fetch optional arguments
content_weight = args.content_weight
style_weight = args.style_weight
tv_weight = args.tv_weight
iter = args.iter


width, height = load_img(content_img_path).size
img_rows = 500
img_cols = int(width * img_rows / height)

# test
# print(args)

def preprocess_img(img_path):
  # load image as JpegImageFile
  img = load_img(img_path, target_size = (img_rows, img_cols))
  # convert JpegImageFile into ndarray(a multidimensional, fixed size array object)
  img = img_to_array(img)

  # reshape 3-dimension img into 4-dimension img (by add a dimension upfront)
  img = np.expand_dims(img, axis=0)
  # preprocess image so as to be compatible for keras
  np_img = preprocess_input(img)
  return np_img

def deprocess_img(np_img):
  np_img = np_img.reshape((img_rows, img_cols, 3))

  # add back testing-mean-pixel (due to default manipulation in openCV from Caffe of Keras)
  np_img[:, :, 0] += 103.939
  np_img[:, :, 1] += 116.779
  np_img[:, :, 2] += 123.68

  # BGR to RGB(due to default manipulation in openCV from Caffe of Keras)
  np_img = np_img[:, :, ::-1]
  img = np.clip(np_img, 0, 255).astype('uint8')
  return img

# get tensor representations of input images
content_img = K.variable(preprocess_img(content_img_path))
style_img = K.variable(preprocess_img(style_img_path))

name = 'name'
i = 0
prefix = name + '_at_iteration_%d.png' % i
# save_img(prefix, deprocess_img(np_img))
# img.show



a = [[1,2,3,9]]
c = [[4,5,6,8],[0,1,2,3]]

inputs = K.placeholder((2, 3))
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