from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from modules import resnet_model
from modules.fpn import fpn
from modules.rpn import rpn, rpn_loss
from modules.pretrained import ImagenetModel

from config import config

flags.DEFINE_integer(
    name='resnet_size', default=50, help='The size of resnet model')
flags.DEFINE_string(
    name='data_format', default='channels_last', help='data format')
flags.DEFINE_integer(name='batch_size', default=1, help='batch size')
flags.DEFINE_integer(name='resnet_version', default=2, help='resnet version')
flags.DEFINE_string(name='mode', default='train', help='train or eval mode')


def resnet_model_fn(model_class, params):

  resnet_size = params['resnet_size']
  data_format = params['data_format']
  resnet_version = params['resnet_version']
  dtype = params['dtype']

  return model_class(
      resnet_size, data_format, resnet_version=resnet_version, dtype=dtype)


def upsnet(features, label=None):

  flags_obj = flags.FLAGS

  num_classes = config.dataset.num_classes 
  num_seg_classes = config.dataset.num_seg_classes 
  num_reg_classes = (2 if config.network.cls_agnostic_bbox_reg else config.dataset.num_classes)

  resnet_backbone = resnet_model_fn(
      ImagenetModel, {
          'dtype': tf.float32,
          'resnet_size': flags_obj.resnet_size,
          'data_format': flags_obj.data_format,
          'batch_size': flags_obj.batch_size,
          'resnet_version': flags_obj.resnet_version
      })

  resnet_branches = resnet_backbone(features, flags_obj.mode == 'train')

  res2 = resnet_branches['res2']
  res3 = resnet_branches['res3']
  res4 = resnet_branches['res4']
  res5 = resnet_branches['res5']

  res2_shape_print = tf.print('res2 shape: ', tf.shape(res2))
  res3_shape_print = tf.print('res3 shape: ', tf.shape(res3))
  res4_shape_print = tf.print('res4 shape: ', tf.shape(res4))
  with tf.control_dependencies(
      [res2_shape_print, res3_shape_print, res4_shape_print]):
    res5_shape_print = tf.print('res5 shape: ', tf.shape(res5))
  # res2: [-1, 80, 80, 256]
  # res3: [-1,

  with_extra_level = True

  if with_extra_level:
    fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = fpn(
        resnet_branches,
        feature_dim=config.network.fpn_feature_dim,
        with_norm=config.network.fpn_with_norm,
        upsample_method=config.network.fpn_upsample_method)
  else:
    fpn_p2, fpn_p3, fpn_p4, fpn_p5 = fpn(
        resnet_branches,
        feature_dim=config.network.fpn_feature_dim,
        with_norm=config.network.fpn_with_norm,
        upsample_method=config.network.fpn_upsample_method)

  rpn_cls_score, rpn_bbox_pred, rpn_cls_prob = [], [], []
  for feat in [fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6]:
    cls_score, bbox_pred, cls_prob = rpn(feat)
    rpn_cls_score.append(cls_score)
    rpn_bbox_pred.append(bbox_pred)
    rpn_cls_prob.append(cls_prob)

  if label is not None:
    pyramid_proposal = PyramidProposal(
        feat_stride=config.network.rpn_feat_stride,
        scales=config.network.anchor_scales,
        ratios=config.network.anchor_ratios,
        rpn_pre_nms_top_n=config.train.rpn_pre_nms_top_n,
        rpn_post_nms_top_n=config.train.rpn_post_nms_top_n,
        threshold=config.train.rpn_nms_thresh,
        rpn_min_size=config.train.rpn_min_size,
        individual_proposals=config.train.rpn_individual_proposals)
    proposal_target = ProposalMaskTarget(
        num_classes=num_reg_classes,
        batch_images=config.train.batch_size,
        batch_rois=config.train.batch_rois,
        fg_fraction=config.train.fg_fraction,
        mask_size=config.network.mask_size,
        binary_thresh=config.network.binary_thresh)

  # if label is not None:
  # RPN loss
  #  rpn_cls_loss, rpn_bbox_loss = rpn_loss(config.train.rpn_batch_size * config.train.batch_size, rpn_cls_score, rpn_bbox_pred, label)

  init = tf.global_variables_initializer()

  with tf.Session() as sess:

    sess.run(init)

    sess.run(res5_shape_print)
    fpn_p2_, fpn_p3_, fpn_p4_, fpn_p5_, fpn_p6_ = sess.run(
        [fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6])
    print(fpn_p2_.shape)
    print(fpn_p3_.shape)
    print(fpn_p4_.shape)
    print(fpn_p5_.shape)
    print(fpn_p6_.shape)

    cls_score, bbox_pred, cls_prob = sess.run(
        [rpn_cls_score[0], rpn_bbox_pred[0], rpn_cls_prob[0]])
    print(cls_score.shape)  # number of anchors = 15
    print(bbox_pred.shape)  # 4 * number of anchors
    print(cls_prob.shape)  # number of anchors = 15