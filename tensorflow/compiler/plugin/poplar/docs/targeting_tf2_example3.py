import tensorflow as tf
from tensorflow.python import ipu

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()


# Create a simple model.
def create_model():
  return tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])


# Create a dataset for the model.
def create_dataset():
  mnist = tf.keras.datasets.mnist

  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.repeat().prefetch(16)


# Define a function which performs a single training step of a model.
def training_step(features, labels, model, optimizer):
  # Execute the model and calculate the loss.
  with tf.GradientTape() as tape:
    predictions = model(features, training=True)
    prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, predictions)
    loss = tf.reduce_mean(prediction_loss)

  # Apply the gradients.
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss


# Create a loop which performs ``steps_per_execution`` iterations of
# ``training_step`` every time this function is executed.
@tf.function(experimental_compile=True)
def training_loop(iterator, steps_per_execution, outfeed, model, optimizer):
  # Create an on device loop.
  for _ in tf.range(steps_per_execution):
    # Get the next input.
    features, labels = next(iterator)

    # Perform the training step.
    loss = training_step(features, labels, model, optimizer)

    # Enqueue the loss after each step to the outfeed queue. This is then read
    # back on the host for monitoring the model performance.
    outfeed.enqueue(loss)


# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Create a Keras model.
  model = create_model()

  # Create an optimizer.
  opt = tf.keras.optimizers.SGD(0.01)

  # Create an iterator inside the strategy for the dataset the model will be
  # trained on.
  iterator = iter(create_dataset())

  # Create an IPUOutfeedQueue to collect results from each step.
  outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

  # Total number of steps (batches) to run.
  total_steps = 100

  # How many steps (batches) to execute each time the device executes.
  steps_per_execution = 10

  for begin_step in range(0, total_steps, steps_per_execution):
    # Run the training loop.
    strategy.run(training_loop,
                 args=(iterator, steps_per_execution, outfeed_queue, model,
                       opt))
    # Calculate the mean loss.
    mean_loss = sum(outfeed_queue) / steps_per_execution
    print(f"Current step: {begin_step}, training loss: {mean_loss}")
