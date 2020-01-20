import math
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


def fcn_subnet(in_channels,
               out_channels,
               num_layers,
               deformable_group=1,
               dilation=1,
               with_norm='none'):

  if with_norm not in ['none', 'batch_norm', 'group_norm']:
    assert ValueError('Invalid normalization type')

  assert num_layers >= 2

  if with_norm == 'batch_norm':
    norm_fn = slim.batch_norm
  elif with_norm == 'group_norm':
    norm_fn = tf.contrib.layers.group_norm
  else:
    norm_fn = None

  for i in range(num_layers):
    if i == num_layers - 2:
      pass
