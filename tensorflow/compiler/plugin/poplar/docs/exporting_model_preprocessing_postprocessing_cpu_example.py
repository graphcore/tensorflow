import os
import shutil

import numpy as np

import tensorflow as tf

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import serving

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/004'
# Directory should be empty or should not exist.
if os.path.exists(saved_model_directory):
  shutil.rmtree(saved_model_directory)


# The preprocessing step is performed fully on the CPU.
@tf.function
def preprocessing_step(x):
  transform_fn = lambda input: tf.constant(
      1.0) if input == "graphcore" else tf.random.uniform(shape=tuple())

  return tf.stack([transform_fn(elem) for elem in tf.unstack(x)])


# The function to export.
@tf.function
def predict_step(x):
  # preprocessing will be compiled and exported together with application body.
  # Double the input - replace this with your application body.
  return x * 2


# The postprocessing step is performed fully on the CPU.
@tf.function
def postprocessing_step(x):
  return tf.abs(x)


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
# Prepare the `postprocessing_step` function signature.
postprocessing_step_signature = (tf.TensorSpec(shape=input_shape,
                                               dtype=np.float32),)

# Export as a SavedModel.
iterations = 10

runtime_func = serving.export_single_step(
    predict_step,
    saved_model_directory,
    iterations,
    predict_step_signature,
    preprocessing_step=preprocessing_step,
    preprocessing_step_signature=preprocessing_step_signature,
    postprocessing_step=postprocessing_step,
    postprocessing_step_signature=postprocessing_step_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
strategy = ipu_strategy.IPUStrategy()
with strategy.scope():
  print(
      runtime_func(
          tf.constant(
              ["graphcore", "red", "blue", "yellow", "graphcore", "purple"],
              dtype=tf.string)))
  print(
      runtime_func(
          tf.constant([
              "apple", "banana", "graphcore", "orange", "pineapple",
              "graphcore"
          ],
                      dtype=tf.string)))
