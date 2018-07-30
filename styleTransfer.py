# # import model VGG19

from PIL import Image
import numpy as np
# import tensorflow as tf

# # helper method for loading images

# convert helper method1: convert np array image to jpeg image
def convert_np_to_jpeg(np_image):
	# rescale pixel-values into range between 0 and 255 and convert to integer
	image = np.clip(np_image,0.0, 255.0)
	image = image.astype(np.uint8)

	# convert to bytes
	jpeg_image = Image.fromarray(image)
	return jpeg_image

# convert helper method2: convert jpeg,jpg,... image to np array image
def convert_jpeg_to_np(jpeg_image):
	np_image = np.float32(jpeg_image)
	return np_image

# display helper method1: display jpeg image
def plot_jpeg_image(jpeg_image):
	jpeg_image.show()

# display helper method2: display np array image
def plot_np_image(np_image):
	# revert to jpeg image and display
	restored_image = convert_np_to_jpeg(np_image)
	restored_image.show()

# resize helper method : calculate resize image factor
def new_size(size, max_size = None):
	if max_size is not None:
		factor = max_size / np.max(size)
		resized = np.array(size) * factor
		resized = resize.astype(int)
	else:
		resized = size
	return resized

def load_image(filename, max_size = None):
	# load jpeg image, resize and return np array image

	image = Image.open(filename)
	image_size = image.size

	resized_size = new_size(image_size, max_size)
	# LANCZOS (a high-quality downsampling filter)
	resized_image = image.resize(resized_size, Image.LANCZOS)

	np_image = convert_jpeg_to_np(resized_image)
	return np_image

def save_image(np_image, filename):
	# rescale pixel-values between 0 and 255
	image = np.clip(np_image,0.0, 255.0)
	
	# convert to bytes
	image = image.astype(np.uint8)
	
	# convert to jpeg file and save
	jpeg = convert_np_to_jpeg(np_image)
	with open(filename, 'wb') as file:
		jpeg.save(file, 'jpeg')

# # def plot_combined_images(content_image, style_image, mixed_image):


# ## loss functions
# performance measurement
# def plot_example_errors:

# mean_squared_error
# def mean_squared_error(old, new):
# 	return tf.reduce_mean(tf.square_difference(old, new))

# # images_placeholder = tf.placeholder(tf.float32, shape= [None, img_size_flat])
# 																		# shape=(batch_size, mnist.IMAGE_PIXELS)


# content loss
def content_loss(session, model, content_image, layer_ids):
	return None
  # feed content_image

  # extract target layer

  # get loss in all targeted layers
def gram_matrix(tensor):
	# helper function for style loss
	
	# extract matrix from tensor
	transposed_matrix * matrix
	transposed = tf.transpose(matrix)
	gram = tf.matmul(transposed, matrix)
	return gram

# style loss
# def style_loss(model, style_image, layer_ids):

# total loss
# def total_loss(weight_content_loss, weight_style_loss, content_loss, style_loss):


# style transfer
def style_transfer(content_image, style_image,content_layer_ids, style_layer_ids,weight_content, weight_style, weight_denoise, num_iterations, strides):
	return None
  #content_loss
  #style_loss

#   for i in range(num_iterations):
    # feed_image

    # calculate gradient, value_adjustment

    # gradient reduce

    # mixed_image_update

#     # plot_combined_image


#   # plot_image(mixed_image)

#   # return mixed_image


# test part
filename = "images/starry_night.jpg"
style_jpeg_image = Image.open(filename)
print(style_jpeg_image.size)
style_np_image = load_image(filename)
print(style_np_image.shape())
# plot_np_image(style_np_image)
# style_jpeg_image = convert_np_to_jpeg(style_np_image)
# new_filename = 'images/new_file.jpeg'
# save_image(style_np_image, new_filename)
# plot_jpeg_image(style_jpeg_image)
