import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

tf.disable_v2_behavior()


def model(batch):
  @ipu.outlined_function
  def func(a):
    with tf.variable_scope("vs", use_resource=True):
      w = tf.get_variable(
          "w",
          shape=[64, 64],
          initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
    x = tf.matmul(a, w)
    return x

  partial = func(batch)
  partial = func(partial)
  # ...
