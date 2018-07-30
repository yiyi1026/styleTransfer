# helper function for creating a new cnn layer
# input tensor
# 1. image number
# 2. y-axis of each image.
# 3. x-axis of each image.
# 4. Channels of each image.

# output tensor
# 1. image number, same as input
# 2. y-axis of each image.
# 3. x-axis of each image.
# 4. Channels produced     s.

# input channels could either be color-channels or filter-channels


def new_conv_layer(prev_layer, 
                  num_input_channels, 
                  filter_size, 
                  num_filters, 
                  use_pooling):
    # return layer, weights


def flatten_layer(layer):
  # 4-dimensional to 2-dimensional
  # return layer_flat, num_features

def new_fullyconnected_layer(prev_layer,
                            num_inputs,
                            num_outputs)
