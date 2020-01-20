from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app as absl_app
from absl import flags
import tensorflow as tf 

from modules.resnet_upsnet import upsnet
from config import config, update_config

FLAGS = flags.FLAGS

flags.DEFINE_string(name='cfg', default=None, help='A path to configuration file')

def main(_):

  if FLAGS.cfg is not None:
    update_config(FLAGS.cfg)

  print(config.dataset.num_classes)
  print(config.dataset.num_seg_classes)

  inputs = tf.random.normal([10, 320, 320, 3])
  upsnet(inputs)

if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  absl_app.run(main)