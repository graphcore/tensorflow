import os
import shutil

import numpy as np

import tensorflow as tf
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import serving
from tensorflow.python.ipu import config

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/001'
# Directory should be empty or should not exist.
if os.path.exists(saved_model_directory):
  shutil.rmtree(saved_model_directory)


# The function to export.
@tf.function
def my_net(x):
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
# Prepare the input signature.
input_signature = (tf.TensorSpec(shape=input_shape, dtype=np.float32),)
# Export as a SavedModel.
iterations = 16

runtime_func = serving.export_single_step(my_net, saved_model_directory,
                                          iterations, input_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
# This should print the even numbers 0 to 30.
strategy = ipu_strategy.IPUStrategy()
with strategy.scope():
  for i in range(iterations):
    input_data = np.ones(input_shape, dtype=np.float32) * i
    print(runtime_func(input_data))
