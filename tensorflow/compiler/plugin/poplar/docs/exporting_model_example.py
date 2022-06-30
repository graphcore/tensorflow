import os
import sys

import numpy as np

import tensorflow as tf
from tensorflow.python.ipu import config
from tensorflow.python.ipu import serving

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/001'

if os.path.exists(saved_model_directory):
  sys.exit(f"Directory '{saved_model_directory}' exists! Please delete it "
           "before running the example.")


# The function to export.
@tf.function
def predict_step(x):
  # Double the input - replace this with application body.
  result = x * 2
  return result


# Configure the IPU for compilation.
cfg = config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.device_connection.enable_remote_buffers = True
cfg.device_connection.type = config.DeviceConnectionType.ON_DEMAND
cfg.configure_ipu_system()

input_shape = (4,)
# Prepare the `predict_step` function signature.
predict_step_signature = (tf.TensorSpec(shape=input_shape, dtype=np.float32),)

# Export as a SavedModel.
iters = 16
runtime_func = serving.export_single_step(
    predict_step,
    saved_model_directory,
    iterations=iters,
    predict_step_signature=predict_step_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
# This should print the even numbers 0 to 30.
input_placeholder = tf.placeholder(dtype=np.float32, shape=input_shape)
result_op = runtime_func(input_placeholder)

with tf.Session() as sess:
  for i in range(iters):
    input_data = np.ones(input_shape, dtype=np.float32) * i
    print(sess.run(result_op, {input_placeholder: input_data}))
