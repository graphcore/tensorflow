import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.keras.datasets import mnist

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 1
config.configure_ipu_system()


# Create a simple model.
def create_model():
  input_layer = tf.keras.layers.Input((28, 28))
  x = tf.keras.layers.Flatten()(input_layer)
  x = tf.keras.layers.Dense(256, activation='relu')(x)
  x = tf.keras.layers.Dense(128, activation='relu')(x)
  x = tf.keras.layers.Dense(10)(x)

  return tf.keras.Model(inputs=input_layer, outputs=x)


# Create a dataset for the model.
def create_dataset():
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
  # Create a Keras model inside the strategy.
  model = create_model()

  # Compile the model for training.
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.RMSprop(),
      metrics=["accuracy"],
  )

  model.fit(dataset, epochs=2, steps_per_epoch=100)
