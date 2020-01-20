"""Deformable convolutions

    Author: Sae Young Kim
    References: https://github.com/kastnerkyle/deform-conv
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
import tensorflow as tf
import tensorflow.contrib.slim as contrib_slim

slim = contrib_slim

def tf_flatten(a):
    """Flatten tensor"""
    return tf.reshape(a, [-1])


def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of np.repeat for 1D"""
    # https://github.com/tensorflow/tensorflow/issues/8521
    assert len(a.get_shape()) == 1

    a = tf.expand_dims(a, -1)
    a = tf.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def tf_repeat_2d(a, repeats):
    """Tensorflow version of np.repeat for 2D"""

    assert len(a.get_shape()) == 2
    a = tf.expand_dims(a, 0)
    a = tf.tile(a, [repeats, 1, 1])
    return a


def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates
    Note that coords is transposed and only 2D is supported
    Parameters
    ----------
    input : tf.Tensor. shape = (s, s)
    coords : tf.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tf.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tf.gather_nd(input, coords_lt)
    vals_rb = tf.gather_nd(input, coords_rb)
    vals_lb = tf.gather_nd(input, coords_lb)
    vals_rt = tf.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = np.array([
        sp_map_coordinates(input, coord.T, mode='nearest', order=1)
        for input, coord in zip(inputs, coords)
    ])
    return mapped_vals


def tf_batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates
    Only supports 2D feature maps
    Parameters
    ----------
    input : tf.Tensor. shape = (b, s, s)
    coords : tf.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_coords = tf.shape(coords)[1]

    coords = tf.clip_by_value(coords, 0, tf.cast(input_size, 'float32') - 1)
    coords_lt = tf.cast(tf.floor(coords), 'int32')
    coords_rb = tf.cast(tf.ceil(coords), 'int32')
    coords_lb = tf.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_rt = tf.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    idx = tf_repeat(tf.range(batch_size), n_coords)

    def _get_vals_by_coords(input, coords):
        indices = tf.stack(
            [idx, tf_flatten(coords[..., 0]),
             tf_flatten(coords[..., 1])],
            axis=-1)
        vals = tf.gather_nd(input, indices)
        vals = tf.reshape(vals, (batch_size, n_coords))
        return vals

    vals_lt = _get_vals_by_coords(input, coords_lt)
    vals_rb = _get_vals_by_coords(input, coords_rb)
    vals_lb = _get_vals_by_coords(input, coords_lb)
    vals_rt = _get_vals_by_coords(input, coords_rt)

    coords_offset_lt = coords - tf.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 1]

    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = np.stack(np.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = np.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def tf_batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input
    Parameters
    ---------
    input : tf.Tensor. shape = (b, s, s)
    offsets: tf.Tensor. shape = (b, s, s, 2)
    """

    input_shape = tf.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    offsets = tf.reshape(offsets, (batch_size, -1, 2))
    grid = tf.meshgrid(tf.range(input_size),
                       tf.range(input_size),
                       indexing='ij')
    grid = tf.stack(grid, axis=-1)
    grid = tf.cast(grid, 'float32')
    grid = tf.reshape(grid, (-1, 2))
    grid = tf_repeat_2d(grid, batch_size)
    coords = offsets + grid

    #print_coords = tf.print(coords)

    #with tf.control_dependencies([print_coords]):
    #  coords = tf.identity(coords)

    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals

def conv2d_offset(inputs, output_channels, kernel_size=3, padding='SAME', use_bias=False, weights_initializer=None, name=None):
        input_shape = inputs.get_shape()
        if weights_initializer is None:
          uniform_unit = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
          # zero_init = tf.zeros_initializer()
          # gaussian_init = tf.random_normal_initializer(mean=0.0, stddev=0.001)
          
        offsets = tf.layers.conv2d(inputs,
                                   filters=output_channels, # output_channels * 2
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   use_bias=use_bias,
                                   kernel_initializer=uniform_unit)

        offsets = tf.transpose(offsets, [0, 3, 1, 2])
        offsets = tf.reshape(offsets,
                             (-1, int(input_shape[1]), int(input_shape[2]), 2))
        inputs = tf.transpose(inputs, [0, 3, 1, 2])
        inputs = tf.reshape(inputs,
                            (-1, int(input_shape[1]), int(input_shape[2])))
        inputs_offset = tf_batch_map_offsets(inputs, offsets)
        inputs_offset = tf.reshape(inputs_offset, (-1, int(
            input_shape[3]), int(input_shape[1]), int(input_shape[2])))
        inputs_offset = tf.transpose(inputs_offset, [0, 2, 3, 1])

        return inputs_offset

def deform_conv_with_offsets(inputs, num_classes=10, batch_norm=True, softmax=False):

    # assert in_channels % groups == 0, 'in_channels must be divisible by groups'
    # assert out_channels % groups == 0, 'out_channels must be divisible by groups'
    # assert out_channels % deformable_groups == 0, 'out_channels must be divisible by deformable groups'

    # stride = (stride for _ in range(2))
    # padding = (padding for _ in range(2))
    # dilation = (dilation for _ in range(2))

  
    # n = in_channels * kernel_size * kernel_size
    # stddev = 1. / (n**0.5)
    # uniform_init = tf.random_uniform_initializer(-stddev, stddev)
    # with tf.variable_scope("deform_conv", reuse=tf.AUTO_REUSE):
    #     w = tf.get_variable("w",
    #                         shape=[
    #                             out_channels, in_channels // groups,
    #                             kernel_size, kernel_size
    #                         ],
    #                         initializer=uniform_init)
    #     b = tf.get_variable("b", w.get_shape()[0], initializer=uniform_init)

    net = tf.reshape(inputs, [-1, 28, 28, 1])

    # deformable cnn
    with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.relu): 
      net = slim.conv2d(net, 32, [3, 3], scope='conv11')
      if batch_norm:
        net = slim.batch_norm(net)
      
      offsets = conv2d_offset(net, 32) 
      net = slim.conv2d(offsets, 64, [3, 3], stride=(2, 2), scope='conv12')
      if batch_norm:
        net = slim.batch_norm(net)

      offsets = conv2d_offset(net, 64) 
      net = slim.conv2d(offsets, 128, [3, 3], stride=(2, 2), scope='conv21')
      if batch_norm:
        net = slim.batch_norm(net)

      offsets = conv2d_offset(net, 128) 
      net = slim.conv2d(offsets, 128, [3, 3], stride=(2, 2), scope='conv22')
      if batch_norm:
       net = slim.batch_norm(net)

    # global average pooling
    net = tf.reshape(slim.avg_pool2d(net, net.get_shape()[1: -1]), [-1, net.get_shape()[-1]])
    net = slim.fully_connected(net, num_classes, activation_fn=None, scope='fc1')
    if softmax:
      net = tf.nn.softmax(net, axis=1)

    return net

if __name__ == '__main__':

  x = tf.random.normal(shape=[10, 784], dtype=tf.float32)
  x = tf.reshape(x, [-1, 28, 28, 1])
  x = deform_conv_with_offsets(x)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    sess.run(init)
    x_ = sess.run(x)
    print('shape:', x_.shape)
    print('tensor:', x_)