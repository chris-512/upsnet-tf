import tensorflow as tf

from config import config


def mask_matching(gt_segs,
                  gt_masks,
                  num_seg_classes,
                  enable_void,
                  class_mapping=None,
                  keep_inds=None):
  """
  :param gt_segs: [1 x h x w]
        :param gt_masks: [num_gt_boxes x h x w]
        :param keep_inds: [num_kept_boxes x 1]
        :return: matched_gt: [1 x h x w]
  """
  if class_mapping is None:
    class_mapping = dict(
        zip(
            range(1, config.dataset.num_classes),
            range(num_seg_classes - config.dataset.num_classes + 1,
                  num_seg_classes)))

  matched_gt = tf.ones_like(gt_segs) * -1
  matched_gt = tf.where(tf.less_equal(gt_segs, config.dataset.num_seg_classes - config.dataset.num_classes), gt_segs, matched_gt)
  matched_gt = tf.where(tf.greater_equal(gt_segs, 255), gt_segs, matched_gt)

  if keep_inds is not None:
    gt_masks = gt_masks[keep_inds]

  for i in range(gt_masks.shape[0]):
    matched_gt[(gt_)]