import tensorflow as tf
from tensorflow.python import ipu

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()


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


dataset = create_dataset()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Using standard keras Sequential model.
  model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  # IPU-specific arguments passed into separate configuration methods.
  model.set_pipeline_stage_assignment([0, 0, 1, 1])

  # Replication factor is 1 in this example.
  model.set_pipelining_options(gradient_accumulation_steps_per_replica=16)

  # steps_per_execution specified to improve performance.
  model.compile(steps_per_execution=256,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer=tf.keras.optimizers.RMSprop())

  model.fit(dataset, epochs=2, steps_per_epoch=128)
