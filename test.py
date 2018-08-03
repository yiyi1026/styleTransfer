import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.applications.vgg19 import VGG19
import argparse

# arguments preparation
parser = argparse.ArgumentParser(description="Style Transfer")

# required arguments
parser.add_argument('content_img_path', metavar='content', type=str, help='path to content image')
parser.add_argument('style_img_path', metavar='style', type=str, help='path to style image')
parser.add_argument('generated_prefix', metavar='gen_prefix', type=str, help='prefix for generated results')

# optional arguments
parse.add_argument("content_weight", type=float, default=0.02, require=False, help="Content Weight")
parse.add_argument("style_weight", type=float, default=0.02, require=False, help="Style Weight")
parse.add_argument("tv_weight", type=float, default=0.02, require=False, help="Total Variation Weight")


args = parser.parse_args()
# print("here are the args")
content_img_path = args.content_img_path
style_img_path = args.style_img_path
generated_prefix = args.generated_prefix



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