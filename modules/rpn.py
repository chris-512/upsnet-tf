from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def rpn(net, num_anchors=15, input_dim=256, with_norm='none', weight_decay=None):

  input_dim = net.get_shape().as_list()[-1]

  def norm_fn(norm_choice):

    assert norm_choice in ['batch_norm', 'group_norm', 'none']

    if norm_choice == 'batch_norm':
      return slim.batch_norm
    elif norm_choice == 'group_norm':
      return None  # TODO(cocoaberry)
    elif norm_choice == 'none':
      return None

  normalize = norm_fn(with_norm)

  with slim.arg_scope(
      [slim.conv2d],
      # By default, he initializaiton.
      # he: tf.contrib.layers.variance_scaling_initializer()
      # - stddev: sqrt(2) / n
      # xavier: tf.contrib.layers.xavier_initializer() or slim.initializers.xavier_initializer()
      # - stddev: 1 / sqrt(n)
      weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), 
      weights_regularizer=None if weight_decay is None else slim.l2_regularizer(weight_decay),
      biases_initializer=tf.zeros_initializer()
  ):
    net = slim.conv2d(net, input_dim, [3, 3], padding='SAME')
    if normalize:
      net = normalize(net)
    net = tf.nn.relu(net)

    cls_score = slim.conv2d(net, num_anchors, [1, 1])
    bbox_pred = slim.conv2d(net, num_anchors * 4, [1, 1])
    cls_prob = tf.nn.sigmoid(cls_score)

    return cls_score, bbox_pred, cls_prob

def rpn_loss(rpn_batch_size, rpn_cls_score, rpn_bbox_pred, label, with_fpn=True):
  
  def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff 
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2)
        loss_box = (in_box_diff ** 2 * (sigma_2 / 2.) * smoothL1_sign 
            + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)) * bbox_outside_weights
        return tf.reduce_sum(loss_box) / tf.shape(loss_box)[0]

  if with_fpn:
    pass
  else:
    pass