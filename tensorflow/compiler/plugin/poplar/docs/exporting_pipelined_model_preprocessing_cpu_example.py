import os
import shutil

import numpy as np
import tensorflow as tf

from tensorflow.python.ipu import config
from tensorflow.python.ipu import ipu_strategy
from tensorflow.python.ipu import serving

# Directory where SavedModel will be written.
saved_model_directory = './my_saved_model_ipu/006'
# Directory should be empty or should not exist.
if os.path.exists(saved_model_directory):
  shutil.rmtree(saved_model_directory)


# The preprocessing stage is performed fully on the CPU.
@tf.function
def preprocessing_step(x):
  transform_fn = lambda input: tf.constant(
      1.0) if input == "graphcore" else tf.random.uniform(shape=tuple())

  return tf.stack([transform_fn(elem) for elem in tf.unstack(x)])


# The pipeline's stages to export.
def stage1(x):
  # Double the input - replace this with 1st stage body.
  output = x * 2
  return output


def stage2(x):
  # Add 3 to the input - replace this with 2nd stage body.
  output = x + 3
  return output


# Configure the IPU for compilation.
cfg = config.IPUConfig()
cfg.auto_select_ipus = 2
cfg.device_connection.enable_remote_buffers = True
cfg.device_connection.type = config.DeviceConnectionType.ON_DEMAND
cfg.configure_ipu_system()

input_shape = (6,)
# Prepare the input signature.
predict_step_signature = (tf.TensorSpec(shape=input_shape, dtype=np.float32),)
# Prepare the `preprocessing_step` function signature.
preprocessing_step_signature = (tf.TensorSpec(shape=input_shape,
                                              dtype=tf.string),)

# Number of times each pipeline stage is executed.
iterations = 10

# Export as a SavedModel.
predict_step = [stage1, stage2]
runtime_func = serving.export_pipeline(
    predict_step,
    saved_model_directory,
    iterations=iterations,
    device_mapping=[0, 1],
    predict_step_signature=predict_step_signature,
    preprocessing_step=preprocessing_step,
    preprocessing_step_signature=preprocessing_step_signature)
print(f"SavedModel written to {saved_model_directory}")

# You can test the exported executable using returned `runtime_func`.
# This should print numbers from 3 to 33.
strategy = ipu_strategy.IPUStrategy()
with strategy.scope():
  print(
      runtime_func(
          tf.constant(["graphcore", "is", "the", "best", "AI", "company"],
                      dtype=tf.string)))
  print(
      runtime_func(
          tf.constant(["make", "new", "AI", "breakthroughs", "with", "IPUS"],
                      dtype=tf.string)))
