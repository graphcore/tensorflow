from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
from tensorflow.python import ipu

step_count = 10000

#
# Configure the IPU system
#
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()


#
# The input data and labels
#
def create_dataset():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat()


#
# The model.  Because this model does not have a specific shape for its inputs
# it will be constructed when it is first called (in the `train` function). So
# it does not need to be an IPU device targeted model.
#
def create_model():
  m = keras.models.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  return m


# The custom training loop
@tf.function
def training_step(features, labels, model, opt):
  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    prediction_loss = keras.losses.sparse_categorical_crossentropy(
        labels, predictions)
    loss = tf.reduce_mean(prediction_loss)

  grads = tape.gradient(loss, model.trainable_variables)
  opt.apply_gradients(zip(grads, model.trainable_variables))
  return loss


# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategyV1()

with strategy.scope():
  # An optimizer for updating the trainable variables
  opt = tf.keras.optimizers.SGD(0.01)

  # Create an instance of the model
  model = create_model()

  # Get the training dataset
  ds = create_dataset()

  # Train the model
  for (x, y), c in zip(ds, range(step_count)):
    loss = strategy.run(training_step, args=[x, y, model, opt])

    if not c % 50:
      print("Step " + str(c) + " loss = " + str(loss.numpy()))
