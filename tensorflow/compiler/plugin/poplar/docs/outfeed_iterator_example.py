# Setup.
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.keras.datasets import mnist

config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()


# Create a dataset.
def create_dataset():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat().prefetch(16)


# Simple model.
def create_model():
  return tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])


@tf.function(experimental_compile=True)
def training_loop(iterator, steps_per_execution, outfeed, model, optimizer):  # pylint: disable=redefined-outer-name
  for _ in tf.range(steps_per_execution):
    # Get the next input.
    features, labels = next(iterator)

    # Execute the model and calculate the loss.
    with tf.GradientTape() as tape:
      predictions = model(features, training=True)
      prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
          labels, predictions)
      loss = tf.reduce_mean(prediction_loss)

    # Apply the gradients.
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Enqueue the results for the outfeed queue.
    outfeed.enqueue(loss)


# Run the loop.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Create a Keras model.
  train_model = create_model()

  # Create an optimizer.
  opt = tf.keras.optimizers.SGD(0.01)

  # Create an iterator for the dataset.
  train_iterator = iter(create_dataset())

  # Create an IPUOutfeedQueue to collect results from each sample.
  outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

  # Total number of steps (batches) to run.
  total_steps = 1000

  # How many steps (batches) to execute each time the device executes.
  steps_per_execution = 10

  for begin_step in range(0, total_steps, steps_per_execution):
    # Run the training loop.
    strategy.run(training_loop,
                 args=(train_iterator, steps_per_execution, outfeed_queue,
                       train_model, opt))
    # Calculate the mean loss.
    total = 0.
    for training_loss in outfeed_queue:
      total += training_loss
    print(f"Loss: {total / steps_per_execution}")
