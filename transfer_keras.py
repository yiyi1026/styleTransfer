import numpy as np
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from keras.layers import InputLayer, Input
# from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
# # , Dropout, Activation

# model = Sequential()
# # import vgg16
# from vgg16 import VGG16
model = VGG19()

# test session
filename = "images/starry_night.jpg"

# load image and save as JpegImageFile
img = load_img(filename)

# transfer JpegImageFile into ndarray, which is multidimensional, fixed size array object
img = img_to_array(img)

# reshapl 3-dimension img into 4-dimension img (the first dimension represent the number of samples), equivalent as img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
img = np.expand_dims(img, axis=0)
# preprocess image so as to be compatible
np_img = preprocess_input(img)





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