import tensorflow as tf
import numpy as np 
from config import config
from rpn.generate_anchors import generate_anchors 
from bbox.bbox_transform import bbox_transform, clip_boxes, bbox_overlaps
from nms.nms import py_nms_wrapper

class PyramidProposal(object):

  def __init__(self, feat_stride, scales, ratios, rpn_pre_nms_top_n,
        rpn_post_nms_top_n, threshold, rpn_min_size, individual_proposals=False, batch_idx=0, use_softnms=False):
    self.feat_stride = feat_stride 
    self.scales = np.array(scales)
    self.ratios = np.array(ratios)
    self.num_anchors = config.network.num_anchors
    self.rpn_pre_nms_top_n = rpn_pre_nms_top_n 
    self.rpn_post_nms_top_n = rpn_post_nms_top_n
    self.threshold = threshold
    self.rpn_min_size = rpn_min_size
    self.individual_proposals = individual_proposals
    self.batch_idx = batch_idx
    self.use_softnms = use_softnms
    self.nms_func = py_nms_wrapper(self.threshold)

  def generate_proposals(self, cls_prob, bbox_pred, im_info):
  
    batch_size = cls_prob[0].shape[0]
    if batch_size > 1:
      raise ValueError("Sorry, multiple images for each device is not implemented.")

    pre_nms_topN = self.rpn_pre_nms_top_n 
    post_nms_topN = self.rpn_post_nms_top_n
    min_size = self.rpn_min_size

    proposal_list = []
    score_list = []
    
    for idx in range(len(self.feat_stride)):
      stride = int(self.feat_stride[idx])
      sub_anchors = generate_anchors(stride=stride, sizes=self.scales * stride, aspect_ratios=self.ratios)

      scores, bbox_deltas = cls_prob[idx], bbox_pred[idx]

      # 1. generate proposals from bbox_deltas and shifted anchors
      # use real image size instead of padded feature map sizes
      height, width = scores.shape[-3:-1]

      # enumerate all shifts 
      shift_x = np.arange(0, width) * stride 
      shift_y = np.arange(0, height) * stride 
      shift_x, shift_y = np.meshgrid(shift_x, shift_y)
      shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

      A = self.num_anchors 
      K = shifts.shape[0]
      anchors = sub_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
      anchors = anchors.reshape((K * A, 4))

      bbox_deltas = bbox_deltas.reshape((-1, 4))
      scores = scores.reshape((-1, 1))

      if self.individual_proposals:
        if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
          order = np.argsort(-scores.squeeze())
        else:
          inds = np.argpartition(
            -scores.squeeze(), pre_nms_topN
          )[:pre_nms_topN]
          order = np.argsort(-scores[inds].squeeze())
          order = inds[order]
        bbox_deltas = bbox_deltas[order, :]
        anchors = anchors[order, :]
        scores = scores[order]
      
      # convert anchors into proposals via bbox transformations
      proposals = bbox_transform(anchors, bbox_deltas)

      # 2. clip predicted boxes to image 
      proposals = clip_boxes(proposals, im_info[:2])

      # 3. remove predicted boxes with either height or width < threshold 
      # (NOTE: convert min_size to input image scale stored in im_info[2])
      # keep = self._filter_boxes(proposals, min_size * im_info[2])
      keep = self._filter_boxes(proposals, min_size * im_info[2])
      proposals = proposals[keep, :]
      scores = scores[keep]

      if self.individual_proposals:
        keep = self.nms_func(np.hstack((proposals, scores)).astype(np.float32))
        if post_nms_topN > 0:
          keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        scores = scores[keep]
      
      proposal_list.append(proposals)
      score_list.append(scores)
  
    proposals = np.vstack(proposal_list)
    scores = np.vstack(score_list)

    batch_inds = np.ones((proposals.shape[0], 1), dtype=np.float32) * self.batch_idx
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
  
  @staticmethod
  def _filter_boxes(boxes, min_size):
      """ Remove all boxes with any side smaller than min_size """
      ws = boxes[:, 2] - boxes[:, 0] + 1
      hs = boxes[:, 3] - boxes[:, 1] + 1
      keep = np.where((ws >= min_size) & (hs >= min_size))[0]
      return keep

  @staticmethod
  def _clip_pad(tensor, pad_shape):
      """
      Clip boxes of the pad area.
      :param tensor: [n, c, H, W]
      :param pad_shape: [h, w]
      :return: [n, c, h, w]
      """
      H, W = tensor.shape[2:]
      h, w = pad_shape

      if h < H or w < W:
          tensor = tensor[:, :, :h, :w].copy()

      return tensor

  def __call__(self, cls_prob, bbox_pred, im_info, roidb=None):

    rois, scores = [], []
    for i in range(im_info.shape[0]):
      rois_im_i, scores_im_i = self.generate_proposals(cls_prob, bbox_pred, im_info[i])
      rois.append(rois_im_i)
      scores.append(scores_im_i)

    # merge
    rois = np.vstack(rois)
    scores = np.vstack(scores)
    idx = np.argsort(-scores, axis=0)
    idx = idx[:self.rpn_post_nms_top_n]

    return rois[idx, :], scores[idx]