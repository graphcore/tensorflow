import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import config
from tensorflow.python.ipu import serving

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/005'

if os.path.exists(saved_model_directory):
  sys.exit(f"Directory '{saved_model_directory}' exists! Please delete it "
           "before running the example.")


# The preprocessing step is performed fully on the IPU.
def preprocessing_step(x):
  return tf.abs(x)


# The postprocessing step is performed fully on the IPU.
def postprocessing_step(x):
  return tf.reduce_sum(x)


def application_body(x):
  # Double the input - replace this with your application body.
  return x * 2


# The function to export.
def predict_step(x):
  # preprocessing and postprocessing will be compiled and exported together
  # with application body.
  x = preprocessing_step(x)
  x = application_body(x)
  return postprocessing_step(x)


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
iters = 10
runtime_func = serving.export_single_step(
    predict_step,
    saved_model_directory,
    iterations=iters,
    predict_step_signature=predict_step_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using the returned `runtime_func`.
input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape)
result_op = runtime_func(input_placeholder)

with tf.Session() as sess:
  for i in range(iters):
    input_data = np.ones(input_shape, dtype=np.float32) * (-1.0 * i)
    print(sess.run(result_op, {input_placeholder: input_data}))
