import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.keras.datasets import mnist

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
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = x_train / 255.0

  train_ds = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(10000).batch(32, drop_remainder=True)
  train_ds = train_ds.map(lambda d, l:
                          (tf.cast(d, tf.float32), tf.cast(l, tf.int32)))

  return train_ds.prefetch(16)


dataset = create_dataset()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Create a Keras model inside the strategy.
  model = create_model()

  # `steps_per_execution` must be divisible by `gradient_accumulation_steps_per_replica`.
  # Say we want to accumulate 10 steps before doing a weight update, then we would end up
  # with the following values.
  gradient_accumulation_steps_per_replica = 10
  number_of_accumulated_steps = dataset.cardinality(
  ) // gradient_accumulation_steps_per_replica

  # In order to get the proper `steps_per_execution` value, we have to multiply
  # `number_of_accumulated_steps` with `gradient_accumulation_steps_per_replica`.
  steps_per_execution = number_of_accumulated_steps * \
                        gradient_accumulation_steps_per_replica

  # Now we need to truncate the dataset so Keras will not try to take more data
  # from the dataset than is available.
  dataset = dataset.take(steps_per_execution)

  # Compile the model for training.
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.RMSprop(),
      metrics=["accuracy"],
      steps_per_execution=steps_per_execution,
  )

  model.set_gradient_accumulation_options(
      gradient_accumulation_steps_per_replica=10)

  model.fit(dataset, epochs=2)
