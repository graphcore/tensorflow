from threading import Thread

from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import ipu_strategy
from tensorflow import keras
import tensorflow as tf

# The host side queue
outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()


# A custom training loop
@tf.function
def training_step(features, labels, in_model, optimizer):
  with tf.GradientTape() as tape:
    predictions = in_model(features, training=True)
    prediction_loss = keras.losses.sparse_categorical_crossentropy(
        labels, predictions)
    loss = tf.reduce_mean(prediction_loss)
    grads = tape.gradient(loss, in_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, in_model.trainable_variables))

  outfeed_queue.enqueue(loss)
  return loss


# Configure the IPU devices
cfg = IPUConfig()
cfg.configure_ipu_system()

# Execute the graph
strategy = ipu_strategy.IPUStrategyV1()
with strategy.scope():
  # Create the dataset for feeding the graphs
  dataset = tf.data.Dataset.from_tensors(tf.constant(1.0, shape=[2, 20]))
  dataset = dataset.repeat()
  # Create the keras model and optimizer
  model = keras.models.Sequential([
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  opt = keras.optimizers.SGD(0.01)
  NUM_ITERATIONS = 100

  # Function to continuously dequeue the outfeed until n examples are seen
  def dequeue_thread_fn():
    counter = 0
    while counter != NUM_ITERATIONS:
      r = outfeed_queue.dequeue().numpy()

      # Check if something has been enqueued
      if r.size:
        # The outfeed may have been enqueued multiple times between dequeues
        for t in r:
          print("Step", counter, "loss = ", t)
          counter += 1

  # Start the dequeuing thread
  dequeue_thread = Thread(target=dequeue_thread_fn)

  # Run the custom training loop over the data.
  for i, (x, y) in zip(range(NUM_ITERATIONS), dataset):
    strategy.run(training_step, args=[x, y, model, opt])
    # Start the dequeue_thread once the graph has been compiled
    if i == 0:
      dequeue_thread.start()

  # Wait for the dequeuing thread to finish
  dequeue_thread.join()
