import argparse
import time

import tensorflow.compat.v1 as tf

from tensorflow.keras import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python import ipu

NUM_CLASSES = 10


def model_fn(features, labels, mode, params):
  """A simple CNN based on https://keras.io/examples/cifar10_cnn/"""

  model = Sequential()
  model.add(Conv2D(16, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(Conv2D(16, (3, 3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(32, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256))
  model.add(Activation("relu"))
  model.add(Dropout(0.5))
  model.add(Dense(NUM_CLASSES))

  logits = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.EVAL:
    predictions = tf.argmax(input=logits, axis=-1)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels,
                                        predictions=predictions),
    }
    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      eval_metric_ops=eval_metric_ops)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(params["learning_rate"])
    if params["replicas"] > 1:
      optimizer = ipu.cross_replica_optimizer.CrossReplicaOptimizer(optimizer)
    train_op = optimizer.minimize(loss=loss)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  raise NotImplementedError(mode)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--test-only",
      action="store_true",
      help="Skip training and test using latest checkpoint from model_dir.")

  parser.add_argument("--batch-size",
                      type=int,
                      default=32,
                      help="The batch size.")

  parser.add_argument(
      "--iterations-per-loop",
      type=int,
      default=100,
      help="The number of iterations (batches) per loop on IPU.")

  parser.add_argument("--log-interval",
                      type=int,
                      default=10,
                      help="Interval at which to log progress.")

  parser.add_argument("--summary-interval",
                      type=int,
                      default=1,
                      help="Interval at which to write summaries.")

  parser.add_argument("--training-steps",
                      type=int,
                      default=200000,
                      help="Total number of training steps.")

  parser.add_argument(
      "--learning-rate",
      type=float,
      default=0.01,
      help="The learning rate used with stochastic gradient descent.")

  parser.add_argument(
      "--replicas",
      type=int,
      default=1,
      help="The replication factor. Increases the number of IPUs "
      "used and the effective batch size by this factor.")

  parser.add_argument(
      "--model-dir",
      help="Directory where checkpoints and summaries are stored.")

  return parser.parse_args()


def create_ipu_estimator(args):
  ipu_options = ipu.config.IPUConfig()
  ipu_options.auto_select_ipus = args.replicas

  ipu_run_config = ipu.ipu_run_config.IPURunConfig(
      iterations_per_loop=args.iterations_per_loop,
      num_replicas=args.replicas,
      ipu_options=ipu_options,
  )

  config = ipu.ipu_run_config.RunConfig(
      ipu_run_config=ipu_run_config,
      log_step_count_steps=args.log_interval,
      save_summary_steps=args.summary_interval,
      model_dir=args.model_dir,
  )

  return ipu.ipu_estimator.IPUEstimator(
      config=config,
      model_fn=model_fn,
      params={
          "learning_rate": args.learning_rate,
          "replicas": args.replicas
      },
  )


def train(ipu_estimator, args, x_train, y_train):
  """Train a model on IPU and save checkpoints to the given `args.model_dir`."""
  def input_fn():
    # If using Dataset.from_tensor_slices(), the data will be embedded
    # into the graph as constants, which makes the training graph very
    # large and impractical. So use Dataset.from_generator() here instead,
    # but add prefetching and caching to improve performance.

    def generator():
      return zip(x_train, y_train)

    types = (x_train.dtype, y_train.dtype)
    shapes = (x_train.shape[1:], y_train.shape[1:])

    dataset = tf.data.Dataset.from_generator(generator, types, shapes)
    dataset = dataset.prefetch(len(x_train)).cache()
    dataset = dataset.repeat()
    dataset = dataset.shuffle(len(x_train))
    dataset = dataset.batch(args.batch_size, drop_remainder=True)

    return dataset

  # Training progress is logged as INFO, so enable that logging level
  tf.logging.set_verbosity(tf.logging.INFO)

  t0 = time.time()
  ipu_estimator.train(input_fn=input_fn, steps=args.training_steps)
  t1 = time.time()

  duration_seconds = t1 - t0
  images_per_step = args.batch_size * args.replicas
  images_per_second = args.training_steps * images_per_step / duration_seconds
  print("Took {:.2f} minutes, i.e. {:.0f} images per second".format(
      duration_seconds / 60, images_per_second))


def calc_batch_size(num_examples, batches_per_loop, batch_size):
  """Reduce the batch size if needed to cover all examples without a remainder."""
  assert batch_size > 0
  assert num_examples % batches_per_loop == 0
  while num_examples % (batch_size * batches_per_loop) != 0:
    batch_size -= 1
  return batch_size


def test(ipu_estimator, args, x_test, y_test):
  """Test the model on IPU by loading weights from the final checkpoint in the
  given `args.model_dir`."""

  num_test_examples = len(x_test)

  batches_per_loop = args.replicas * args.iterations_per_loop
  test_batch_size = calc_batch_size(num_test_examples, batches_per_loop,
                                    args.batch_size)

  if test_batch_size != args.batch_size:
    print("Test batch size changed to {}.".format(test_batch_size))

  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(test_batch_size, drop_remainder=True)
    return dataset

  num_steps = num_test_examples // (test_batch_size * args.replicas)
  metrics = ipu_estimator.evaluate(input_fn=input_fn, steps=num_steps)
  test_loss = metrics["loss"]
  test_accuracy = metrics["accuracy"]

  print("Test loss: {:g}".format(test_loss))
  print("Test accuracy: {:.2f}%".format(100 * test_accuracy))


def main():
  args = parse_args()
  train_data, test_data = cifar10.load_data()

  num_test_examples = len(test_data[0])
  batches_per_loop = args.replicas * args.iterations_per_loop
  if num_test_examples % batches_per_loop != 0:
    raise ValueError(("replicas * iterations_per_loop ({} * {}) must evenly " +
                      "divide the number of test examples ({})").format(
                          args.replicas, args.iterations_per_loop,
                          num_test_examples))

  ipu_estimator = create_ipu_estimator(args)

  def normalise(x, y):
    return x.astype("float32") / 255.0, y.astype("int32")

  if not args.test_only:
    print("Training...")
    x_train, y_train = normalise(*train_data)
    train(ipu_estimator, args, x_train, y_train)

  print("Testing...")
  x_test, y_test = normalise(*test_data)
  test(ipu_estimator, args, x_test, y_test)


if __name__ == "__main__":
  main()
