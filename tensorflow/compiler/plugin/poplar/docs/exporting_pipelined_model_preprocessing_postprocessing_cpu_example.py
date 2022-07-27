import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import config
from tensorflow.python.ipu import serving

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/006'

if os.path.exists(saved_model_directory):
  sys.exit(f"Directory '{saved_model_directory}' exists! Please delete it "
           "before running the example.")


# The preprocessing stage is performed fully on the CPU.
def preprocessing_step(x):
  # The preprocessing stage is performed on the cpu.
  return tf.abs(x)


# The pipeline stages to export.
def stage1(x):
  # Double the input - replace this with 1st stage body.
  output = x * 2
  return output


def stage2(x):
  # Add 3 to the input - replace this with 2nd stage body.
  output = x + 3
  return output


# The postprocessing step is performed fully on the CPU.
def postprocessing_step(x):
  return tf.abs(x)


# Configure the IPU for compilation.
cfg = config.IPUConfig()
cfg.auto_select_ipus = 2
cfg.device_connection.enable_remote_buffers = True
cfg.device_connection.type = config.DeviceConnectionType.ON_DEMAND
cfg.configure_ipu_system()

input_shape = (4,)
# Prepare the input signature.
predict_step_signature = (tf.TensorSpec(shape=input_shape, dtype=np.float32),)
# Prepare the `preprocessing_step` function signature.
preprocessing_step_signature = (tf.TensorSpec(shape=input_shape,
                                              dtype=tf.float32),)
# Prepare the `postprocessing_step` function signature.
postprocessing_step_signature = (tf.TensorSpec(shape=input_shape,
                                               dtype=np.float32),)

# Export as a SavedModel.
iters = 10
predict_step = [stage1, stage2]
runtime_func = serving.export_pipeline(
    predict_step,
    saved_model_directory,
    iterations=iters,
    device_mapping=[0, 1],
    predict_step_signature=predict_step_signature,
    preprocessing_step=preprocessing_step,
    preprocessing_step_signature=preprocessing_step_signature,
    postprocessing_step=postprocessing_step,
    postprocessing_step_signature=postprocessing_step_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`,
input_placeholder = tf.placeholder(dtype=tf.float32, shape=input_shape)
result_op = runtime_func(input_placeholder)

with tf.Session() as sess:
  for i in range(iters):
    input_data = np.ones(input_shape, dtype=np.float32) * (-1.0 * i)
    print(sess.run(result_op, {input_placeholder: input_data}))
