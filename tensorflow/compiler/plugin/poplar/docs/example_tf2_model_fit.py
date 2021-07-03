import tensorflow as tf
from tensorflow import keras
from tensorflow.python import ipu

#
# Configure the IPU system.
#
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 1
cfg.configure_ipu_system()

#
# Create the training and evaluation datasets.
#

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Categorize.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Wrap data in TF Dataset.
training_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
training_data = training_data.batch(8, drop_remainder=True)

eval_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
eval_data = eval_data.batch(8, drop_remainder=True)


#
# Create a CNN model using the IPU-specific Model class.
#
def create_model():
  # Create an input node to specify input properties.
  inputs = keras.Input(shape=(32, 32, 3), name="img")

  # Block 1.
  x = keras.layers.Conv2D(8, 3, activation="relu")(inputs)
  x = keras.layers.Conv2D(16, 3, activation="relu")(x)
  block_1_output = keras.layers.MaxPooling2D(3)(x)

  # Block 2.
  x = keras.layers.Conv2D(16, 3, activation="relu",
                          padding="same")(block_1_output)
  x = keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)
  block_2_output = keras.layers.add([x, block_1_output])

  # Block 3.
  x = keras.layers.Conv2D(16, 3, activation="relu")(block_2_output)
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dense(64, activation="relu")(x)
  # Use IPU specific dropout for improved performance.
  x = ipu.keras.layers.Dropout(0.5)(x)
  x = keras.layers.Dense(10)(x)

  # Create a Keras model.
  return keras.Model(inputs=inputs, outputs=x)


# Create an IPU distribution strategy.
strategy = ipu.ipu_strategy.IPUStrategyV1()

with strategy.scope():
  model = create_model()

  # Print a model summary.
  model.summary()

  # Compile the model, configuring a loss, optimizer and metric.
  model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=keras.optimizers.RMSprop(1e-3),
                metrics=["accuracy"],
                steps_per_execution=32)

  # Train for two epochs.
  model.fit(x_train, y_train, batch_size=8, epochs=2, steps_per_epoch=128)

  # Evaluate trained model.
  test_scores = model.evaluate(x_test, y_test, batch_size=8, steps=128)
  print(f"Test loss: {test_scores[0]}")
  print(f"Test accuracy: {test_scores[1]}")
