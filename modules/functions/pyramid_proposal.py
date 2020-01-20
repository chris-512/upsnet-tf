import numpy as np 

class PyramidProposalFunction(object):

  def __init__(self,
               feat_stride,
               scales,
               ratios,
               rpn_pre_nms_top_n,
               rpn_post_nms_top_n,
               threshold,
               rpn_min_size,
               individual_proposals=False,
               batch_idx=0,
               use_softnms=False,
               crowd_gt_roi=None):
    super(PyramidProposalFunction, self).__init__()
    self.feat_stride = feat_stride
    self.scales = np.array(scales)
    self.ratios = np.array(ratios)
    self.num_anchors = 3
    self.rpn_pre_nms_top_n = rpn_pre_nms_top_n
    self.rpn_post_nms_top_n = rpn_post_nms_top_n
    self.threshold = threshold
    self.rpn_min_size = rpn_min_size
    self.individual_proposals = individual_proposals
    self.batch_idx = batch_idx
    self.use_softnms = use_softnms
    self.crowd_gt_roi = crowd_gt_roi

def __call__(self, cls_probs, bbox_preds, im_info):

  assert len(cls_prob) == 6
  assert len(bbox_pred) == 6

  nms = gpu_nms_wrapper(self.threshold, device_id=device_id)
  
  pre_nms_topN = self.rpn_pre_nms_top_n 
  post_nms_topN = self.rpn_post_nms_top_n 
  min_size = self.rpn_min_size 

  proposal_list = []
  score_list = []
  im_info = im_info.numpy() 

  for i in range(len(self.feat_stride)):
    stride = int(self.feat_stride[i])
    sub_anchors = generate_anchors(stride=stride, size=self.scales * stride, aspect_ratios=self.ratios)
    scores = cls_probs[i]
    bbox_deltas = bbox_preds[i]

    height, width = scores.shape[-2:]

    