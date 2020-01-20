# 머신러닝 학습의 Hello World 와 같은 MNIST(손글씨 숫자 인식) 문제를 신경망으로 풀어봅니다.
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from modules.deform_conv import deform_conv_with_offsets

flags = tf.app.flags

flags.DEFINE_string("eval_dir", "./logs",
                    "Directory where summaries are saved.")
flags.DEFINE_integer("batch_size", 50, "batch_size")

FLAGS = flags.FLAGS

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

batch_size = FLAGS.batch_size
total_batch = int(mnist.train.num_examples / batch_size)
max_steps = 15 * total_batch
validation_interval = 50
eval_dir = FLAGS.eval_dir

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# deformable cnn
y_ = deform_conv_with_offsets(x)

loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

is_correct = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

tf.summary.scalar('accuracy', accuracy)

summary = tf.summary.merge_all()

init = tf.global_variables_initializer()

test_writer = tf.summary.FileWriter(eval_dir)

with tf.Session() as sess:

  sess.run(init)

  for iter in range(max_steps):

    batch_xs, batch_ys = mnist.train.next_batch(batch_size)

    _, loss_ = sess.run([optimizer, loss],
                        feed_dict={
                            x: batch_xs,
                            y: batch_ys
                        })

    print('iter-{} loss = {}'.format(iter, loss_))

    if iter % validation_interval == 0:
      summary_, test_acc = sess.run([summary, accuracy],
                                    feed_dict={
                                        x: mnist.test.images[:1000],
                                        y: mnist.test.labels[:1000]
                                    })
      print('test accuracy: {}'.format(test_acc))
      test_writer.add_summary(summary_, global_step=iter)
      test_writer.flush()
