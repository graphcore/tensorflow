import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

tf.disable_v2_behavior()


def model(batch):
  @ipu.outlined_function
  def func(lhs, rhs):
    x = tf.matmul(lhs, rhs)
    return x

  # Create the variables.
  with tf.variable_scope("vs", use_resource=True):
    w1 = tf.get_variable(
        "w1",
        shape=[64, 64],
        initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
    w2 = tf.get_variable(
        "w2",
        shape=[64, 64],
        initializer=tf.glorot_uniform_initializer(dtype=tf.float32))

  # Pass the variables as inputs to the function.
  partial = func(batch, w1)
  partial = func(partial, w2)
  # ...
