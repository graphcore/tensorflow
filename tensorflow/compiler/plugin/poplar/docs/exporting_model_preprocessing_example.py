import os
import shutil

import numpy as np

import tensorflow as tf

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import serving

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/003'
# Directory should be empty or should not exist.
if os.path.exists(saved_model_directory):
  shutil.rmtree(saved_model_directory)


# The preprocessing step is performed fully on the IPU.
def preprocessing_step(x):
  return tf.abs(x)


def application_body(x):
  # Double the input - replace this with your application body.
  return x * 2


# The function to export.
@tf.function
def predict_step(x):
  # preprocessing will be compiled and exported together with application body.
  x = preprocessing_step(x)

  return application_body(x)


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
iterations = 10

runtime_func = serving.export_single_step(predict_step, saved_model_directory,
                                          iterations, predict_step_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
# This should print the even numbers 0 to 30.
strategy = ipu_strategy.IPUStrategy()
with strategy.scope():
  for i in range(iterations):
    input_data = np.ones(input_shape, dtype=np.float32) * (-1.0 * i)
    print(runtime_func(input_data))
