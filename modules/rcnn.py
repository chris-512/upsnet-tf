import tensorflow as tf 

class RCNNLoss(object):

    def __init__(self):
        pass 
    
    def __call__(self, cls_score, bbox_pred, cls_label, bbox_target, bbox_weight):
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(logits=cls_score, labels=cls_label)
        bbox_loss = tf.losses.huber_loss(bbox_target * bbox_weight, bbox_pred * bbox_weight, reduction=tf.losses.Reduction.SUM)
        return cls_loss, bbox_loss

class MaskRCNNLoss(object):

    def __init__(self, mask_size):
        self.mask_size = mask_size 

    def rcnn_accuracy(self, cls_score, cls_label):
        cls_pred = tf.argmax(cls_score, axis=1)
        ignore = tf.reduce_sum(tf.cast(cls_label == -1, dtype=tf.float32))
        correct = tf.reduce_sum(tf.cast(tf.reshape(cls_pred, [-1]), dtype=tf.float32)) - ignore 
        total = tf.shape(tf.reshape(cls_label, [-1]))[0] - ignore

        return correct / total
        
    def mask_loss(self, input, target, weight):
        binary_input = tf.where(tf.greater_equal(input, 0), 1.0, 0.0)
        loss = -input * (target - binary_input) + tf.log(1 + tf.exp(input - 2 * input * binary_input))
        loss = loss * weight
        return tf.reduce_sum(loss)
    
    def smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff 
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2)
        loss_box = (in_box_diff ** 2 * (sigma_2 / 2.) * smoothL1_sign 
            + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)) * bbox_outside_weights
        return tf.reduce_sum(loss_box) / tf.shape(loss_box)[0]

    
    def __call__(self, cls_score, bbox_pred, mask_score, cls_label, bbox_target, bbox_inside_weight, bbox_outside_weight, mask_target):
        cls_loss = tf.nn.softmax_cross_entropy_with_logits(logits=cls_score, labels=cls_label)
        bbox_loss = self.smooth_l1_loss(bbox_pred, bbox_target, bbox_inside_weight, bbox_outside_weight)
        rcnn_acc = self.rcnn_accuracy(cls_score, cls_label)
        mask_target = tf.reshape(mask_target, [-1, self.mask_size, self.mask_size])
        mask_weight = tf.where(tf.not_equal(mask_target, -1.0), 1.0, 0.0)
        mask_loss = self.mask_loss(mask_score, mask_target, mask_weight) / (tf.reduce_sum(mask_weight) + 1e-10)

        return cls_loss, bbox_loss, mask_loss, rcnn_acc 