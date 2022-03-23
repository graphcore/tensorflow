import os
import numpy as np

from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Configure argument for targeting the IPU
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

with tf.device("cpu"):
  x_data = tf.placeholder(np.float32, [4])


def add_op(x, y):
  outputs = {
      "output_types": [tf.float32],
      "output_shapes": [tf.TensorShape([4])],
  }

  base_path = os.path.join(os.getcwd(), "tensorflow/compiler/plugin/poplar")
  lib_path = os.path.join(base_path, "libcustom_add_inplace.so")

  o = ipu.custom_ops.precompiled_user_op([x, y], lib_path, outs=outputs)
  return o


def my_net(x):
  inplace = tf.get_variable("weights",
                            shape=[4],
                            initializer=tf.zeros_initializer())

  # Even though the custom op is in place, TF still needs to see an assignment.
  inplace_add = tf.assign(inplace, add_op(inplace, x)[0])
  with tf.control_dependencies([inplace_add]):
    return inplace


with ipu_scope("/device:IPU:0"):
  xla_result = ipu.ipu_compiler.compile(my_net, [x_data])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  result = sess.run(xla_result, feed_dict={x_data: [2., 4., 6., -1.]})
  print(result)

  result = sess.run(xla_result, feed_dict={x_data: [2., 4., 6., -1.]})
  print(result)
