from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import config


def fpn(backbone_feats,
        feature_dim,
        with_extra_level=True,
        with_norm='none',
        upsample_method='nearest',
        weight_decay=None):

  assert len(backbone_feats) == 4
  assert upsample_method in ['nearest', 'bilinear']

  def norm_fn(norm_choice):

    assert norm_choice in ['batch_norm', 'group_norm', 'none']

    if norm_choice == 'batch_norm':
      return slim.batch_norm
    elif norm_choice == 'group_norm':
      return None  # TODO(cocoaberry)
    elif norm_choice == 'none':
      return None

  def fpn_upsample(input, scale_factor=2):
    scaled_shape = tf.shape(input)[1:-1] * scale_factor
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if upsample_method == 'nearest' else tf.image.ResizeMethod.BILINEAR
    return tf.image.resize(input, scaled_shape, method=method)

  normalize = norm_fn(with_norm)

  with slim.arg_scope(
      [slim.conv2d],
      # By default, he initializaiton.
      # he: tf.contrib.layers.variance_scaling_initializer()
      # - stddev: sqrt(2) / n
      # xavier: tf.contrib.layers.xavier_initializer() or slim.initializers.xavier_initializer()
      # - stddev: 1 / sqrt(n)
      weights_initializer=tf.contrib.layers.variance_scaling_initializer(), 
      weights_regularizer=None if weight_decay is None else slim.l2_regularizer(weight_decay),
      biases_initializer=tf.zeros_initializer()
  ):
    with tf.variable_scope('fpn_p5_1x1'):
      fpn_p5_1x1 = slim.conv2d(backbone_feats['res5'], feature_dim,
                               [1, 1])  # The number of input channels: 2048
      if normalize:
        fpn_p5_1x1 = normalize(fpn_p5_1x1)
    with tf.variable_scope('fpn_p4_1x1'):
      fpn_p4_1x1 = slim.conv2d(backbone_feats['res4'], feature_dim,
                               [1, 1])  # The number of input channels: 1024
      if normalize:
        fpn_p4_1x1 = normalize(fpn_p4_1x1)
    with tf.variable_scope('fpn_p3_1x1'):
      fpn_p3_1x1 = slim.conv2d(backbone_feats['res3'], feature_dim,
                               [1, 1])  # The number of input channels: 512
      if normalize:
        fpn_p3_1x1 = normalize(fpn_p3_1x1)
    with tf.variable_scope('fpn_p2_1x1'):
      fpn_p2_1x1 = slim.conv2d(backbone_feats['res2'], feature_dim,
                               [1, 1])  # The number of input channels: 256
      if normalize:
        fpn_p2_1x1 = normalize(fpn_p2_1x1)

    if config.network.fpn_with_gap:
      fpn_gap = tf.reshape(
          slim.fully_connected(
              tf.squeeze(slim.avg_pool2d(backbone_feats[3], [1, 1])),
              feature_dim), [-1, feature_dim, 1, 1])
      fpn_p5_1x1 = fpn_p5_1x1 + fpn_gap

    fpn_p5_upsample = fpn_upsample(fpn_p5_1x1)
    fpn_p4_plus = fpn_p5_upsample + fpn_p4_1x1
    fpn_p4_upsample = fpn_upsample(fpn_p4_1x1)
    fpn_p3_plus = fpn_p4_upsample + fpn_p3_1x1
    fpn_p3_upsample = fpn_upsample(fpn_p3_1x1)
    fpn_p2_plus = fpn_p3_upsample + fpn_p2_1x1

    fpn_p5 = slim.conv2d(fpn_p5_1x1, feature_dim, [3, 3], padding='SAME')
    if normalize:
      fpn_p5 = normalize(fpn_p5)
    fpn_p4 = slim.conv2d(fpn_p4_plus, feature_dim, [3, 3], padding='SAME')
    if normalize:
      fpn_p4 = normalize(fpn_p4)
    fpn_p3 = slim.conv2d(fpn_p3_plus, feature_dim, [3, 3], padding='SAME')
    if normalize:
      fpn_p3 = normalize(fpn_p3)
    fpn_p2 = slim.conv2d(fpn_p2_plus, feature_dim, [3, 3], padding='SAME')
    if normalize:
      fpn_p2 = normalize(fpn_p2)

  if with_extra_level:
    fpn_p6 = slim.max_pool2d(fpn_p5, kernel_size=[1, 1], stride=2)
    return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6
  else:
    return fpn_p2, fpn_p3, fpn_p4, fpn_p5
