import itertools
import math
import os
import shutil

import numpy as np
import tensorflow as tf
from tensorflow.python import ipu


def create_tensorboard_log_dir():
  """Handles deletion and creation of the TensorBoard log directory.
    """
  # Ensure we have the top level /tmp directory first.
  path = "/tmp"
  if not os.path.isdir(path):
    raise RuntimeError(
        "Unable to locate %s directory. Are you running on a Unix-like OS?" %
        path)

  # Delete the log directory, if it exists.
  path += "/tensorboard_example_logs"
  if os.path.isdir(path):
    try:
      shutil.rmtree(path)
    except OSError as e:
      print("Unable to remove %s due to: %s" % (e.filename, e.strerror))

  # Create the log directory.
  try:
    os.mkdir(path)
  except OSError as e:
    print("Unable to create %s due to: %s" % (e.filename, e.strerror))

  return path


class StaggeredCallback(tf.keras.callbacks.Callback):
  """A simple `tf.keras.callbacks.Callback` derived class that
    provides a check to allow staggered execution every `period` batches.

    Args:
      period (int): The number of iterations that should elapse before
      `StaggeredCallback.should_run` returns `True`.
    """
  def __init__(self, period):
    self._period = period
    self._n = 0

  def should_run(self):
    self._n += 1
    return (self._n - 1) % self._period == 0


class EvaluateCallback(StaggeredCallback):
  """Runs an evaluation pass on the provided test dataset.

    Args:
      path (str): The directory to which evaluation logs will be written.
      model (tf.keras.Model): The Model on which to run evaluation.
      pred_dataset (tf.Dataset): A Dataset to use for prediction.
      period (int): The number of epochs to elapse before evaluating.
      steps (int): The number of steps to perform evaluation in.
    """
  def __init__(self, path, model, pred_dataset, period, steps=128):
    super().__init__(period)
    self._model = model
    self._ds = pred_dataset
    self._writer = tf.summary.create_file_writer(path)
    self._steps = steps

  def on_epoch_end(self, epoch, logs=None):  # pylint: disable=unused-argument
    if not self.should_run():
      return

    res = self._model.evaluate(self._ds, steps=self._steps, return_dict=True)

    with self._writer.as_default():
      for k, v in res.items():
        tf.summary.scalar("evaluation_%s" % k, v, step=epoch)


class FilterRenderCallback(StaggeredCallback):
  """A `StaggeredCallback` derived class that renders the kernels of a
    given list of convolution layers.

    Args:
      path (string): The directory to which rendered filters will be written.
      model (tf.keras.Model): The Model from which to extract filters.
      layer_names (list(str)): A list of layer names from which filters should
      be extracted and rendered.
      period (int): The number of epochs to elapse before rendering filters.
    """
  def __init__(self, path, model, layer_names, period):
    super().__init__(period)
    self._model = model
    self._layer_names = layer_names
    self._writer = tf.summary.create_file_writer(path)

  def tile_filters(self, filters, name, step):
    # First normalize the filters.
    fmin = tf.reduce_min(filters)
    fmax = tf.reduce_max(filters)
    filters_norm = (filters - fmin) / (fmax - fmin)

    # Instead of providing each filter as a separate image to summary_ops.image
    # we here collapse the leading dimensions to yield a (D, D, N) tensor,
    # where D is the filter size and N the total number of filters.
    shape = (filters_norm.shape[0], filters_norm.shape[1],
             filters_norm.shape[2] * filters_norm.shape[3])
    filters_collapsed = tf.reshape(filters_norm, shape)

    # Split into a list of N DxD filters.
    filters_split = tf.experimental.numpy.dsplit(filters_collapsed,
                                                 filters_collapsed.shape[-1])

    # Find the width and height of the resulting image, not in pixels but
    # in the number of filters to display in each dimension.
    WH = int(math.sqrt(len(filters_split)))

    # Generate each row of the output image.
    filter_num = 0
    rows = []
    for _ in range(WH):
      # Collect the fiters for this row.
      row_images = [
          tf.squeeze(t) for t in filters_split[filter_num:filter_num + WH]
      ]

      # Generate vertical padding to be used between filters.
      col_padding = [tf.zeros((filters.shape[0], 1))] * WH

      # Interleave filters and padding.
      row = list(itertools.chain(*zip(row_images, col_padding)))

      # Stack, excluding the last padding tensor as there is no fillter
      # to the right of it.
      rows.append(tf.experimental.numpy.hstack(row[:-1]))

      filter_num += WH

    # Find the width of the image.
    W = rows[-1].shape[1]

    # Generate horizontal padding.
    row_padding = [tf.zeros((1, W))] * WH
    padded_rows = list(itertools.chain(*zip(rows, row_padding)))

    # Stack and expand dims.
    img = tf.experimental.numpy.vstack(padded_rows[:-1])
    img = tf.expand_dims(tf.expand_dims(img, -1), 0)

    # Write out the resultant image.
    with self._writer.as_default():
      tf.summary.image(name, img, step=step)

  def on_epoch_end(self, epoch, logs=None):  # pylint: disable=unused-argument
    if not self.should_run():
      return

    for layer_name in self._layer_names:
      filters = self._model.get_layer(layer_name).get_weights()[0]
      self.tile_filters(filters, layer_name, epoch)


def create_datasets(train_validate_split=0.8, num_to_prefetch=16):
  """Create datasets for training, validation and evaluation.

    Args:
      train_validate_split (float): The proportion of data to use for training
      versus evaluation.
      num_to_prefetch (int): The number of dataset elements that
      should be prefetched.

    Returns:
      tuple(tf.Dataset): Training, validation and testing datasets.
    """
  def prepare_ds(x, y):
    # Normalize.
    x = np.expand_dims(x, -1) / 255.0

    # Shuffle and batch.
    ds = tf.data.Dataset.from_tensor_slices(
        (x, y)).shuffle(10000).batch(32, drop_remainder=True)

    # Cast and convert targets to one-hot representation.
    ds = ds.map(lambda d, l:
                (tf.cast(d, tf.float32), tf.cast(tf.one_hot(l, 10), tf.int32)))

    return ds

  # Load MNIST dataset. The training set is split into training and
  # validation sets.
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  N = int(train_validate_split * x_train.shape[0])

  # Get the datasets.
  training_ds = prepare_ds(x_train[:N], y_train[:N])
  validation_ds = prepare_ds(x_train[N:], y_train[N:])
  test_ds = prepare_ds(x_test, y_test)

  # Repeat (for training), prefetch and return.
  return (training_ds.repeat().prefetch(num_to_prefetch),
          validation_ds.prefetch(num_to_prefetch),
          test_ds.prefetch(num_to_prefetch))


def create_model():
  """Create a simple CNN to be split into two pipeline stages.
    """
  return tf.keras.Sequential([
      # Input layers do not get assigned an IPU pipeline stage.
      tf.keras.layers.Input((28, 28, 1)),

      # Pipeline stage 0.
      tf.keras.layers.Conv2D(32,
                             kernel_size=(3, 3),
                             activation="relu",
                             name="conv_0"),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_0"),
      tf.keras.layers.Conv2D(64,
                             kernel_size=(3, 3),
                             activation="relu",
                             name="conv_1"),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_1"),

      # Pipeline stage 1.
      tf.keras.layers.Conv2D(32,
                             kernel_size=(3, 3),
                             activation="relu",
                             name="conv_2"),
      tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_2"),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(10, activation="softmax"),
  ])


# Setup log path.
log_path = create_tensorboard_log_dir()

# Configure the IPU device.
config = ipu.config.IPUConfig()
config.auto_select_ipus = 2
config.configure_ipu_system()

# Create a strategy for execution on the IPU.
strategy = ipu.ipu_strategy.IPUStrategy()
with strategy.scope():
  # Create a Keras model inside the strategy.
  m = create_model()
  training_dataset, validation_dataset, test_dataset = create_datasets()

  # Metrics.
  metrics = [
      "accuracy",
      tf.keras.metrics.AUC(),
      tf.keras.metrics.TopKCategoricalAccuracy(3)
  ]

  # Compile the model for training.
  m.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.RMSprop(),
            metrics=metrics,
            steps_per_execution=256)

  # Setup pipelining.
  m.set_pipelining_options(gradient_accumulation_steps_per_replica=16)
  m.set_pipeline_stage_assignment([0] * 4 + [1] * 5)

  # Trigger callbacks every steps_per_epoch steps.
  m.set_asynchronous_callbacks(True)

  # Create callbacks for the training session.
  callbacks = [
      FilterRenderCallback(log_path, m, ['conv_%d' % n for n in [0, 1, 2]], 2),
      EvaluateCallback(log_path, m, test_dataset, 1),
      tf.keras.callbacks.TensorBoard(log_path, histogram_freq=1)
  ]

  # Train.
  m.fit(training_dataset,
        epochs=10,
        steps_per_epoch=128,
        validation_data=validation_dataset,
        validation_freq=2,
        validation_steps=128,
        callbacks=callbacks,
        verbose=False)
