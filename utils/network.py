"""CNN model architecture."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv_act_layer(layer_name, input_tensor, input_dim, output_dim,
                    conv_kernel=[3, 3],
                    conv_strides=[1, 1],
                    conv_padding='SAME',
                    act=tf.nn.relu):
    """A convolutional layer + an activation layer.

    Args:
      layer_name: scope name for the layer
      input_tensor: an input tensor with the dimensions (N_examples, width, height, channel).
      input_dim: number of input feature maps
      output_dim: number of output feature maps
      conv_kernel: size of convolutional kernel
      conv_strides: stride for the convolution
      conv_padding: padding for the convolution
      act: activation layer used

    Returns:
      tensor of shape (N_examples, width, height, channel)

    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(conv_kernel + [input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('preactivations'):
            preactivate = tf.nn.conv2d(input_tensor, weights,
                                       strides=[1] + conv_strides + [1],
                                       padding=conv_padding) + biases
            tf.summary.histogram('preactivations', preactivate)
        with tf.name_scope('activations'):
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations


def max_pool_layer(layer_name,
                   input_tensor,
                   pool_kernel=[1, 2, 2, 1],
                   pool_strides=[1, 2, 2, 1],
                   pool_padding='SAME'):
    """max_pooling layer.

    Args:
      layer_name: scope name for the layer
      input_tensor: an input tensor with the dimensions (N_examples, width, height, channel).
      pool_kernel: size of pooling kernel
      pool_strides: stride for the pooling
      pool_padding: padding for the pooling

    Returns:
      tensor of shape (N_examples, width, height, channel)

    """
    with tf.name_scope(layer_name):
        pool = tf.nn.max_pool(value=input_tensor,
                              ksize=pool_kernel,
                              strides=pool_strides,
                              padding=pool_padding,
                              name=layer_name)
        tf.summary.histogram('pool', pool)
        return pool


def fc_act_layer(layer_name, input_tensor, input_dim, output_dim, act=tf.nn.relu):
    """A fully connected layer + an activation layer.

    Args:
      layer_name: scope name for the layer
      input_tensor: an input tensor with the dimensions (N_examples, input_dim)
      input_dim: number of input neurons
      output_dim: number of output neurons
      act: activation layer used

    Returns:
      tensor of shape (N_examples, output_dim)

    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('preactivations'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('preactivations', preactivate)
        with tf.name_scope('activations'):
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
        return activations


def cnn(x, num_output_c, num_output_r):
    """Network for combined classification and regression. 5 conv layers are shared between classification and regression tasks
    Separate fc layers for individual task.
    2.5D image patch as input

    Args:
      x: an input tensor with the dimensions (N_examples, width, height, channel).
      num_output_c: number of output neurons in the classification layer. (before softmax) (6 neurons)
      num_output_r: number of output neurons in the regression layer. (1 neuron)

    Returns:
      A tuple (yc, yr, keep_prob).
      yc is a tensor of shape (N_examples, num_output_c).
      yr is a tensor of shape (N_examples, num_output_r).
      keep_prob is a scalar placeholder for the probability of dropout.
    """
    ####### Shared Layers #######
    # First convolution block
    conv1_1 = conv_act_layer(layer_name='conv1_1',
                              input_tensor=x,
                              input_dim=30,
                              output_dim=32,
                              conv_kernel=[3, 3],
                              conv_strides=[1, 1],
                              conv_padding='VALID',
                              act=tf.nn.relu)
    pool1 = max_pool_layer(layer_name='pool1',
                           input_tensor=conv1_1,
                           pool_kernel=[1, 2, 2, 1],
                           pool_strides=[1, 2, 2, 1],
                           pool_padding='VALID')

    # Second convolution block
    conv2_1 = conv_act_layer(layer_name='conv2_1',
                              input_tensor=pool1,
                              input_dim=32,
                              output_dim=64,
                              conv_kernel=[3, 3],
                              conv_strides=[1, 1],
                              conv_padding='VALID',
                              act=tf.nn.relu)
    pool2 = max_pool_layer(layer_name='pool2',
                           input_tensor=conv2_1,
                           pool_kernel=[1, 2, 2, 1],
                           pool_strides=[1, 2, 2, 1],
                           pool_padding='VALID')

    # Third convolution block
    conv3_1 = conv_act_layer(layer_name='conv3_1',
                              input_tensor=pool2,
                              input_dim=64,
                              output_dim=128,
                              conv_kernel=[3, 3],
                              conv_strides=[1, 1],
                              conv_padding='VALID',
                              act=tf.nn.relu)
    pool3 = max_pool_layer(layer_name='pool3',
                           input_tensor=conv3_1,
                           pool_kernel=[1, 2, 2, 1],
                           pool_strides=[1, 2, 2, 1],
                           pool_padding='VALID')

    # Fourth convolution block
    conv4_1 = conv_act_layer(layer_name='conv4_1',
                             input_tensor=pool3,
                             input_dim=128,
                             output_dim=256,
                             conv_kernel=[3, 3],
                             conv_strides=[1, 1],
                             conv_padding='VALID',
                             act=tf.nn.relu)
    pool4 = max_pool_layer(layer_name='pool4',
                           input_tensor=conv4_1,
                           pool_kernel=[1, 2, 2, 1],
                           pool_strides=[1, 2, 2, 1],
                           pool_padding='VALID')

    # Fifth convolution block
    conv5_1 = conv_act_layer(layer_name='conv5_1',
                             input_tensor=pool4,
                             input_dim=256,
                             output_dim=512,
                             conv_kernel=[3, 3],
                             conv_strides=[1, 1],
                             conv_padding='VALID',
                             act=tf.nn.relu)
    pool5 = max_pool_layer(layer_name='pool5',
                           input_tensor=conv5_1,
                           pool_kernel=[1, 2, 2, 1],
                           pool_strides=[1, 2, 2, 1],
                           pool_padding='VALID')

    # Reshape final convolution layer
    pre_fc_dim = pool5.get_shape().as_list()[1:]
    fc_input_dim = pre_fc_dim[0] * pre_fc_dim[1] * pre_fc_dim[2]
    pool5_flat = tf.reshape(pool5, [-1, fc_input_dim])


    ####### Classification Layers #######
    # Fully connected layer, fc1_c
    fc1_c = fc_act_layer(layer_name='fc1_c',
                         input_tensor=pool5_flat,
                         input_dim=fc_input_dim,
                         output_dim=1024,
                         act=tf.nn.relu)

    # Dropout layer
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        drop1_c = tf.nn.dropout(fc1_c, keep_prob)

    # Fully connected layer, fc2_c
    fc2_c = fc_act_layer(layer_name='fc2_c',
                         input_tensor=drop1_c,
                         input_dim=1024,
                         output_dim=1024,
                         act=tf.nn.relu)

    # Dropout layer
    with tf.name_scope('dropout'):
        drop2_c = tf.nn.dropout(fc2_c, keep_prob)

    # Classification layer
    yc = fc_act_layer(layer_name='output_c',
                      input_tensor=drop2_c,
                      input_dim=1024,
                      output_dim=num_output_c,
                      act=tf.identity)


    ####### Regression Layers #######
    # Fully connected layer, fc1_r
    fc1_r = fc_act_layer(layer_name='fc1_r',
                         input_tensor=pool5_flat,
                         input_dim=fc_input_dim,
                         output_dim=1024,
                         act=tf.nn.relu)

    # Dropout layer
    with tf.name_scope('dropout'):
        drop1_r = tf.nn.dropout(fc1_r, keep_prob)

    # Fully connected layer, fc2_r
    fc2_r = fc_act_layer(layer_name='fc2_r',
                         input_tensor=drop1_r,
                         input_dim=1024,
                         output_dim=1024,
                         act=tf.nn.relu)

    # Dropout layer
    with tf.name_scope('dropout'):
        drop2_r = tf.nn.dropout(fc2_r, keep_prob)

    # Regression layer
    yr = fc_act_layer(layer_name='output_r',
                      input_tensor=drop2_r,
                      input_dim=1024,
                      output_dim=num_output_r,
                      act=tf.identity)

    return yc, yr, keep_prob