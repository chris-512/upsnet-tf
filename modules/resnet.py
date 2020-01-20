import tensorflow as tf
import tensorflow.contrib.slim as slim

from modules.deform_conv import conv2d_offset
from config import config

def bottleneck(inputs,
               planes,
               stride=1,
               dilation=1,
               downsample=None,
               fix_bn=True,
               expansion=4):

    with tf.variable_scope('plain', reuse=tf.AUTO_REUSE):

        residual = inputs

        with tf.variable_scope('conv1'):
            net = slim.conv2d(inputs, planes, [1, 1], stride=stride)
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)

        with tf.variable_scope('conv2'):
            net = slim.conv2d(net,
                              planes, [3, 3],
                              stride=1,
                              padding=dilation,
                              rate=dilation)
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)

        with tf.variable_scope('conv3'):
            net = slim.conv2d(net, planes * expansion, kernel_size=1)
            net = slim.batch_norm(net)

        if downsample is not None:
            residual = downsample(inputs)

        net += residual
        net = tf.nn.relu(net)

        return net


def deformable_bottleneck(inputs,
                          planes,
                          stride=1,
                          dilation=1,
                          downsample=None,
                          fix_bn=True,
                          deformable_group=1,
                          expansion=4):

    with tf.variable_scope('dcn', reuse=tf.AUTO_REUSE):

        residual = inputs

        with tf.variable_scope('conv1'):
            net = slim.conv2d(inputs, planes, [1, 1], stride=stride)
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)

        with tf.variable_scope('conv2'):
            net = conv2d_offset(net,
                                18 * deformable_group,
                                kernel_size=3,
                                padding='SAME',
                                weights_initializer=tf.zeros_initializer())
            net = slim.conv2d(net,
                              planes, [3, 3],
                              stride=1,
                              padding=dilation,
                              rate=dilation)
            net = slim.batch_norm(net)
            net = tf.nn.relu(net)

        with tf.variable_scope('conv3'):
            net = slim.conv2d(net, planes * expansion, kernel_size=1)
            net = slim.batch_norm(net)

        if downsample is not None:
            residual = downsample(inputs)

        net += residual
        net = tf.nn.relu(net)

        return net


def res_block(net,
              output_channels,
              blocks,
              block=bottleneck,
              stride=1,
              dilation=1,
              fix_bn=True,
              with_dpyramid=False):

    if block.__name__ == 'bottleneck':
        expansion_rate = 4
    elif block.__name__ == 'deformable_bottleneck':
        expansion_rate = 4
    else:
        raise ValueError('Invalid block function')

    def _downsample_fn(net):
        net = slim.conv2d(net,
                          output_channels * expansion_rate, [1, 1],
                          stride=stride)
        net = slim.batch_norm(net)
        return net

    downsample = None
    if stride != 1:
        downsample = _downsample_fn

    net = block(net, output_channels, stride, dilation, downsample, fix_bn)
    slim.repeat(net,
                blocks - 2,
                block,
                planes=output_channels,
                dilation=dilation,
                fix_bn=fix_bn)

    if with_dpyramid:
        net = deformable_bottleneck(net,
                                    output_channels,
                                    dilation=dilation,
                                    fix_bn=fix_bn)
    else:
        net = bottleneck(net,
                         output_channels,
                         dilation=dilation,
                         fix_bn=fix_bn)

    return net


def resnet_backbone(net, blocks):

    fix_bn = config.network.backbone_fix_bn
    with_dilation = config.network.backbone_with_dilation
    with_dpyramid = config.network.backbone_with_dconv
    with_dconv = config.network.backbone_with_dconv

    net_dict = {}

    net = slim.conv2d(net, 64, [7, 7], stride=2, padding="SAME", scope='conv1')
    net = slim.batch_norm(net)
    net = tf.nn.relu(net)
    net = slim.max_pool2d(net, [3, 3], stride=2, padding="SAME", scope='pool1')

    with tf.name_scope('res2'):
        net = res_block(net, 64, blocks[0], fix_bn=fix_bn)
        net_dict['res2'] = net

    with tf.name_scope('res3'):
        net = res_block(
            net,
            128,
            blocks[1],
            block=deformable_bottleneck if with_dconv <= 3 else bottleneck,
            stride=2,
            fix_bn=fix_bn,
            with_dpyramid=with_dpyramid)
        net_dict['res3'] = net

    with tf.name_scope('res4'):
        net = res_block(
            net,
            256,
            blocks[2],
            block=deformable_bottleneck if with_dconv <= 4 else bottleneck,
            stride=2,
            fix_bn=fix_bn,
            with_dpyramid=with_dpyramid)
        net_dict['res4'] = net

    if with_dilation:
        res5_stride, res5_dilation = 1, 2
    else:
        res5_stride, res5_dilation = 2, 1

    with tf.name_scope('res5'):
        net = res_block(
            net,
            512,
            blocks[3],
            block=deformable_bottleneck if with_dconv <= 4 else bottleneck,
            stride=res5_stride,
            dilation=res5_dilation,
            fix_bn=fix_bn)
        net_dict['res5'] = net

    return {
        'res2': net_dict['res2'], 
        'res3': net_dict['res3'], 
        'res4': net_dict['res4'], 
        'res5': net_dict['res5']
    }
