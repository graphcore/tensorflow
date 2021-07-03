from tensorflow.python.ipu.config import IPUConfig

import tensorflow as tf

from tensorflow import keras
from tensorflow.python import ipu

#
# Configure the IPU system
#
cfg = IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()


#
# Create the input data and labels
#
def create_dataset():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.float32)))

  return train_ds.repeat()


#
# Create the model using the IPU-specific Sequential class
#
def create_model():
  m = keras.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  return m


# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategyV1()

with strategy.scope():
  # Create an instance of the model
  model = create_model()

  # Get the training dataset
  ds = create_dataset()

  # Train the model
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                steps_per_execution=100)
  model.fit(ds, steps_per_epoch=2000, epochs=4)
