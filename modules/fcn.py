import tensorflow as tf 
import tensorflow.contrib.slim as slim 

def fcn_subnet(net, out_channels, num_layers, deformable_group=1, dilation=1, with_norm='none'):

  assert with_norm in ['none', 'batch_norm', 'group_norm']
  assert num_layers >= 2

  if with_norm == 'batch_norm':
    norm_fn = slim.batch_norm
  else:
    norm_fn = None

  in_channels = net.get_shape().as_list()[0]
  for i in range(num_layers):
    if i == num_layers - 2:
      slim.conv2d(net, out_channels, [3, 3], stride=(1, 1), padding='SAME')
      in_channels = out_channels
    else:
      slim.conv2d(net, in_channels, [3, 3], stride=(1, 1), padding='SAME')
    if with_norm != 'none':
     net = norm_fn(net)  
    net = tf.nn.relu(net)
  return net 

class FCNHead(object):

    def __init__(self, num_classes, num_layers, with_norm='none', with_roi_loss=False, upsample_rate=4):
        self.upsample_rate = upsample_rate
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.with_norm = with_norm
        self.scale_factor = [2, 4, 8]

    def __call__(self, fpn_p2, fpn_p3, fpn_p4, fpn_p5, roi=None):
        fpn_p2 = fcn_subnet(fpn_p2, 128, num_layers=self.num_layers, with_norm=self.with_norm)
        fpn_p3 = fcn_subnet(fpn_p3, 128, num_layers=self.num_layers, with_norm=self.with_norm)
        fpn_p4 = fcn_subnet(fpn_p4, 128, num_layers=self.num_layers, with_norm=self.with_norm)
        fpn_p5 = fcn_subnet(fpn_p5, 128, num_layers=self.num_layers, with_norm=self.with_norm)

        method = tf.image.ResizeMethod.BILINEAR
        scaled_shape = tf.shape(fpn_p3)[1:-1] * self.scale_factor[0]
        fpn_p3 = tf.image.resize(fpn_p3, scaled_shape, method=method)
        scaled_shape = tf.shape(fpn_p4)[1:-1] * self.scale_factor[1]
        fpn_p4 = tf.image.resize(fpn_p4, scaled_shape, method=method)
        scaled_shape = tf.shape(fpn_p5)[1:-1] * self.scale_factor[2]
        fpn_p5 = tf.image.resize(fpn_p5, scaled_shape, method=method)
        feat = tf.concat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], axis=-1) # [N, 80, 80, 1024]
        score = slim.conv2d(feat, self.num_classes, [1, 1]) # [N, 80, 80, 133]
        outputs = {
          'fcn_score': score,
          'fcn_feat': feat
        }
        if self.upsample_rate != 1:
          scaled_shape = tf.shape(score)[1:-1] * self.upsample_rate
          output = tf.image.resize(score, scaled_shape, method=method) # [N, 320, 320, 133]
          outputs.update({'fcn_output': output})
        #if roi is not None:
        #  roi_feat = self.roipool(feat, roi)
        #  roi_score = slim.conv2d(roi_feat, self.num_classes, [1, 1])
        #  outputs.update({'fcn_roi_score': roi_score})

        return outputs