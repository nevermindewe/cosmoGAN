from pprint import pprint
import sys

import tensorflow as tf


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, transpose_b=False):
    """Make and return a matrix multiply (input_ * w) + b(ias) operation.
    
    """

    shape = input_.get_shape().as_list()
    if not transpose_b:
        w_shape = [shape[1], output_size]
    else:
        w_shape = [output_size, shape[1]]

    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable('w', w_shape, tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('b', [output_size],
                               initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix, transpose_b=transpose_b) + bias


def conv2d(input_, out_channels, data_format, kernel=5, stride=2, stddev=0.02, name="conv2d"):
    """Make and return a 2D convolution + bias operation.
    """
    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        strides = [1, stride, stride, 1]
    else:  # NCHW
        # WARNING: These strides are probably broken
        #   https://www.tensorflow.org/api_docs/python/tf/nn/conv2d:
        #    "Must have strides[0] = strides[3] = 1"
        in_channels = input_.get_shape()[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        # By default, our convolutional mask is a 5x5 filter.
        w = tf.get_variable('w', [kernel, kernel, in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME', data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases, data_format=data_format), conv.get_shape())

        return conv


def conv2d_transpose(input_, output_shape, data_format, kernel=5, stride=2, stddev=0.02,
                     name="conv2d_transpose"):

    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        out_channels = output_shape[-1]
        strides = [1, stride, stride, 1]
    else:
        # WARNING: These strides are probably broken
        #   https://www.tensorflow.org/api_docs/python/tf/nn/conv2d:
        #    "Must have strides[0] = strides[3] = 1"
        in_channels = input_.get_shape()[1]
        out_channels = output_shape[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, out_channels, in_channels],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=strides, data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases, data_format=data_format), deconv.get_shape())

        return deconv


def lrelu(x, alpha=0.2, name="lrelu"):
    with tf.name_scope(name):
        return tf.maximum(x, alpha*x)


def average_gradients(tower_grads):
    """Taken from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

    Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.
    
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    pprint(tower_grads, sys.stderr)
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            print >>sys.stderr, "var name:", _.name, _
            if g is None:
                pass
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    
    return average_grads
    

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
