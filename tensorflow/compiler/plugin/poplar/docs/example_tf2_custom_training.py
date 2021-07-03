from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras
from tensorflow.python import ipu

step_count = 1000
steps_per_execution = 10

#
# Configure the IPU system.
#
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()


#
# The input data and labels.
#
def create_dataset():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat()


#
# The model. Because this model does not have a specific shape for its inputs
# it will be constructed when it is first called (in the `training_step`
# function).
#
def create_model():
  m = keras.models.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  return m


#
# The custom training loop.
#
@tf.function(experimental_compile=True)
def training_loop(iterator, outfeed_queue, model, optimizer, num_iterations):
  for _ in tf.range(num_iterations):
    # Get the data for the step.
    features, labels = next(iterator)

    # Perform the training step.
    with tf.GradientTape() as tape:
      predictions = model(features, training=True)
      prediction_loss = keras.losses.sparse_categorical_crossentropy(
          labels, predictions)
      loss = tf.reduce_mean(prediction_loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Store the loss in the outfeed queue.
    outfeed_queue.enqueue(loss)


# Create an IPU distribution strategy
strategy = ipu.ipu_strategy.IPUStrategyV1()

with strategy.scope():
  # An optimizer for updating the trainable variables.
  opt = tf.keras.optimizers.SGD(0.01)

  # Create an instance of the model.
  model = create_model()

  # Create an iterator for the dataset.
  iterator = iter(create_dataset())

  # Create an IPUOutfeedQueue to collect results from each on device step.
  outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

  # Train the model.
  for step_begin in range(0, step_count, steps_per_execution):

    # Run `steps_per_execution` at a time.
    strategy.run(
        training_loop,
        args=[iterator, outfeed_queue, model, opt, steps_per_execution])

    # Get results for each step.
    for step, loss in zip(range(step_begin, step_begin + steps_per_execution),
                          outfeed_queue):
      if step % 50 == 0:
        print(f"Step {step}: loss {loss}")
