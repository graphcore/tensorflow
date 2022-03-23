import tensorflow.compat.v1 as tf
from tensorflow.python import ipu

tf.disable_v2_behavior()


def model(batch):
  # The outer function is just a Python function.
  def func(a, variable_scope_name):
    # The inner function is an IPU function which captures the variable scope
    # name using Python closures to create scopes.
    @ipu.outlined_function
    def f(a):
      with tf.variable_scope(variable_scope_name, use_resource=True):
        w = tf.get_variable(
            "w",
            shape=[64, 64],
            initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
      x = tf.matmul(a, w)
      return x

    return f(a)

  partial = func(batch, "block1")
  partial = func(partial, "block2")
  # ...
