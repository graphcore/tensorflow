from threading import Thread

from tensorflow.python import ipu
import tensorflow as tf

NUM_ITERATIONS = 100

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
      (x_train, y_train)).shuffle(10000)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))
  train_ds = train_ds.batch(32, drop_remainder=True)
  return train_ds.repeat()


#
# The host side queue
#
outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()


#
# A custom training loop
#
@tf.function(experimental_compile=True)
def training_step(num_iterations, iterator, in_model, optimizer):

  for _ in tf.range(num_iterations):
    features, labels = next(iterator)
    with tf.GradientTape() as tape:
      predictions = in_model(features, training=True)
      prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)
      loss = tf.reduce_mean(prediction_loss)
      grads = tape.gradient(loss, in_model.trainable_variables)
      optimizer.apply_gradients(zip(grads, in_model.trainable_variables))

    outfeed_queue.enqueue(loss)


#
# Execute the graph
#
strategy = ipu.ipu_strategy.IPUStrategyV1()
with strategy.scope():
  # Create the Keras model and optimizer.
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  opt = tf.keras.optimizers.SGD(0.01)

  # Create an iterator for the dataset.
  train_iterator = iter(create_dataset())

  # Function to continuously dequeue the outfeed until NUM_ITERATIONS examples
  # are seen.
  def dequeue_thread_fn():
    counter = 0
    while counter != NUM_ITERATIONS:
      for loss in outfeed_queue:
        print("Step", counter, "loss = ", loss.numpy())
        counter += 1

  # Start the dequeuing thread.
  dequeue_thread = Thread(target=dequeue_thread_fn, args=[])
  dequeue_thread.start()

  # Run the custom training loop over the dataset.
  strategy.run(training_step,
               args=[NUM_ITERATIONS, train_iterator, model, opt])
  dequeue_thread.join()
