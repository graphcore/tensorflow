import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import config
from tensorflow.python.ipu import serving

# Directory where SavedModel will be written
saved_model_directory = './my_saved_model_ipu/004'

if os.path.exists(saved_model_directory):
  sys.exit(f"Directory '{saved_model_directory}' exists! Please delete it "
           "before running the example.")


def preprocessing_step(x):
  def transform_fn(inp):
    is_gc = lambda: tf.constant(1.0)
    is_oth = lambda: tf.random.uniform(shape=[])
    condition = tf.equal(inp, tf.constant("Graphcore", dtype=tf.string))
    return tf.cond(condition, is_gc, is_oth)

  return tf.stack([transform_fn(elem) for elem in tf.unstack(x)])


# The function to export.
def predict_step(x):
  # Preprocessing will be compiled and exported together with
  # the application body.
  # Double the input - replace this with your application body.
  return x * 2


# Configure the IPU for compilation.
cfg = config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.device_connection.enable_remote_buffers = True
cfg.device_connection.type = config.DeviceConnectionType.ON_DEMAND
cfg.configure_ipu_system()

input_shape = (6,)
# Prepare the `predict_step` function signature.
predict_step_signature = (tf.TensorSpec(shape=input_shape, dtype=np.float32),)

# Prepare the `preprocessing_step` function signature.
preprocessing_step_signature = (tf.TensorSpec(shape=input_shape,
                                              dtype=tf.string),)

# Export as a SavedModel.
iters = 10
runtime_func = serving.export_single_step(
    predict_step,
    saved_model_directory,
    iterations=iters,
    predict_step_signature=predict_step_signature,
    preprocessing_step=preprocessing_step,
    preprocessing_step_signature=preprocessing_step_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
input_placeholder = tf.placeholder(dtype=tf.string, shape=input_shape)
result_op = runtime_func(input_placeholder)

with tf.Session() as sess:
  input_data = ["make", "AI", "breakthroughs", "with", "Graphcore", "IPUS"]
  print(sess.run(result_op, {input_placeholder: input_data}))
