import os
import json
import numpy as np

from tensorflow.python import ipu
from tensorflow.python.ipu.scopes import ipu_scope
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Configure argument for targeting the IPU
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

base_path = os.path.join(os.getcwd(), "tensorflow/compiler/plugin/poplar")
lib_path = os.path.join(base_path, "libtutorial_attributes_example.so")


def my_net(x, y):
  x_shape = x.get_shape().as_list()
  y_shape = y.get_shape().as_list()
  outputs = {
      "output_types": [x.dtype],
      "output_shapes": [tf.TensorShape([x_shape[0], y_shape[1]])],
  }

  # We create a matmul operation, which we want to perform as two serialized
  # matmuls. We also record all the input shapes.
  attributes = {
      "serialization_factor": 2,
      "lhs_shape": x_shape,
      "rhs_shape": y_shape
  }
  attributes_json = json.dumps(attributes)

  o = ipu.custom_ops.precompiled_user_op([x, y],
                                         lib_path,
                                         attributes=attributes_json,
                                         outs=outputs)

  return o


with tf.device("cpu"):
  x_ph = tf.placeholder(np.float32, [128, 1024])
  y_ph = tf.placeholder(np.float32, [1024, 64])

with ipu_scope("/device:IPU:0"):
  xla_result = ipu.ipu_compiler.compile(my_net, [x_ph, y_ph])

with tf.Session() as sess:
  # Base run
  result = sess.run(xla_result,
                    feed_dict={
                        x_ph: np.full(x_ph.shape, 10.0),
                        y_ph: np.full(y_ph.shape, 12.0),
                    })

  print(result)
