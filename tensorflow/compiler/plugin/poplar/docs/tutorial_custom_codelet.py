import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import os

from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope

# Configure argument for targeting the IPU
cfg = ipu.utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
cfg = ipu.utils.auto_select_ipus(cfg, 1)
ipu.utils.configure_ipu_system(cfg)

with tf.device("cpu"):
  p_data = tf.placeholder(np.float32, [4, 2])
  p_angle = tf.placeholder(np.float32, [4])


def my_net(x, y):
  outputs = {
      "output_types": [tf.float32],
      "output_shapes": [tf.TensorShape([4, 2])],
  }
  lib_path = os.getcwd() + "/tensorflow/python/ipu/librotate.so"
  gp_path = os.getcwd() + "/"

  x = ipu.internal_ops.precompiled_user_op([x, y],
                                           "Rotate",
                                           lib_path,
                                           gp_path,
                                           outs=outputs)
  return x


with ipu_scope("/device:IPU:0"):
  xla_result = ipu.ipu_compiler.compile(my_net, [p_data, p_angle])

with tf.Session() as sess:
  # Base run
  result = sess.run(
      xla_result,
      feed_dict={
          p_data: [[1., 1.], [2., 0.], [2., 0.], [2., 0.]],
          p_angle: [0., np.pi, np.pi / 2., -np.pi]
      })

  print(result)
