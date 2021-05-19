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
  y_data = tf.placeholder(np.float32, [4])
  p_angle = tf.placeholder(np.float32, [4])


def rotate_op(x, y, a):
  outputs = {
      "output_types": [tf.float32, tf.float32],
      "output_shapes": [tf.TensorShape([4]),
                        tf.TensorShape([4])],
  }

  base_path = os.path.join(os.getcwd(), "tensorflow/compiler/plugin/poplar")
  lib_path = os.path.join(base_path, "libcustom_rotate_op.so")
  gp_path = os.path.join(base_path, "custom_codelet.gp")

  o = ipu.custom_ops.precompiled_user_op([x, y, a],
                                         lib_path,
                                         gp_path,
                                         outs=outputs)
  return o


def my_net(x, y, a):
  return rotate_op(x, y, a)


with ipu_scope("/device:IPU:0"):
  xla_result = ipu.ipu_compiler.compile(my_net, [x_data, y_data, p_angle])

with tf.Session() as sess:
  # Base run
  result = sess.run(xla_result,
                    feed_dict={
                        x_data: [2., 4., 6., -1.],
                        y_data: [2., 3., 8., -1.],
                        p_angle: [np.pi, np.pi / 2., 3. * np.pi / 2., 0]
                    })

  print(result)
