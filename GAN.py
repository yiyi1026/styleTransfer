# use keras for Generative Adversarial Network
from keras import layers
from keras.layers import Dense, Conv2D

# Conv2D
# input shape for "channels_last" is (batch, rowls, cols, channels)
# output shape for "channels_last" is (batch, new_rowls, new_cols, filters), rows and cols might have changed due to padding

layer = Dense(32)
config = layer.get_config()

print(config)

# def discriminator(images)
#   # takes in an images and return a scalar number between 0 to 1 to describe whether the input images is real or fake
#   # use CNN



# def generator(z, batch_size, z_dim)