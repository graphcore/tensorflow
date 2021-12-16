import argparse
import time

import tensorflow.compat.v1 as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python import ipu
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga

NUM_CLASSES = 10


def model_fn(mode, params):
  """A simple CNN based on https://keras.io/examples/cifar10_cnn/ split
  into two pipeline stages placed on different IPUs."""

  # Tell the dropout layers whether we are training to avoid a placeholder.
  is_training = mode == tf.estimator.ModeKeys.TRAIN

  def stage1(features, labels):
    partial = Conv2D(16, (3, 3), padding="same")(features)
    partial = Activation("relu")(partial)
    partial = Conv2D(16, (3, 3))(partial)
    partial = Activation("relu")(partial)
    partial = MaxPooling2D(pool_size=(2, 2))(partial)
    partial = Dropout(0.25)(partial, training=is_training)

    return partial, labels

  def stage2(partial, labels):
    partial = Conv2D(32, (3, 3), padding="same")(partial)
    partial = Activation("relu")(partial)
    partial = Conv2D(32, (3, 3))(partial)
    partial = Activation("relu")(partial)
    partial = MaxPooling2D(pool_size=(2, 2))(partial)
    partial = Dropout(0.25)(partial, training=is_training)

    partial = Flatten()(partial)
    partial = Dense(256)(partial)
    partial = Activation("relu")(partial)
    partial = Dropout(0.5)(partial, training=is_training)
    logits = Dense(NUM_CLASSES)(partial)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
      # This return value is passed to the `optimizer_function`.
      return loss

    if mode == tf.estimator.ModeKeys.EVAL:
      predictions = tf.argmax(input=logits, axis=1, output_type=tf.int32)
      # These return values are passed to the `eval_metrics_fn`.
      return loss, predictions, labels

    raise NotImplementedError(mode)

  def optimizer_function(loss):
    optimizer = tf.train.GradientDescentOptimizer(params["learning_rate"])
    return ipu.pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

  def eval_metrics_fn(loss, predictions, labels):
    # This is executed on the host.
    return {
        "loss": loss,
        "accuracy": tf.metrics.accuracy(predictions=predictions,
                                        labels=labels),
    }

  return ipu.ipu_pipeline_estimator.IPUPipelineEstimatorSpec(
      mode,
      computational_stages=[stage1, stage2],
      optimizer_function=optimizer_function,
      eval_metrics_fn=eval_metrics_fn,
      gradient_accumulation_count=params["gradient_accumulation_count"],
      reduction_method=ga.GradientAccumulationReductionMethod.SUM)


def parse_args():
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--test-only",
      action="store_true",
      help="Skip training and test using latest checkpoint from model_dir.")

  parser.add_argument("--batch-size",
                      type=int,
                      default=16,
                      help="The batch size.")

  parser.add_argument(
      "--gradient-accumulation-count",
      type=int,
      default=4,
      help="The the number of batches that will be pipelined together.")

  parser.add_argument(
      "--iterations-per-loop",
      type=int,
      default=100,
      help="The number of iterations (batches consumed) per loop on IPU.")

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
                      default=100000,
                      help="Total number of training steps.")

  parser.add_argument(
      "--learning-rate",
      type=float,
      default=0.01,
      help="The learning rate used with stochastic gradient descent.")

  parser.add_argument(
      "--model-dir",
      help="Directory where checkpoints and summaries are stored.")

  return parser.parse_args()


def create_ipu_estimator(args):
  num_ipus_in_pipeline = 2

  ipu_options = ipu.config.IPUConfig()
  ipu_options.auto_select_ipus = num_ipus_in_pipeline

  ipu_run_config = ipu.ipu_run_config.IPURunConfig(
      num_shards=num_ipus_in_pipeline,
      iterations_per_loop=args.iterations_per_loop,
      ipu_options=ipu_options,
  )

  config = ipu.ipu_run_config.RunConfig(
      ipu_run_config=ipu_run_config,
      log_step_count_steps=args.log_interval,
      save_summary_steps=args.summary_interval,
      model_dir=args.model_dir,
  )

  return ipu.ipu_pipeline_estimator.IPUPipelineEstimator(
      config=config,
      model_fn=model_fn,
      params={
          "learning_rate": args.learning_rate,
          "gradient_accumulation_count": args.gradient_accumulation_count,
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
  images_per_second = args.training_steps * args.batch_size / duration_seconds
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

  test_batch_size = calc_batch_size(num_test_examples,
                                    args.iterations_per_loop, args.batch_size)

  if test_batch_size != args.batch_size:
    print("Test batch size changed to {}.".format(test_batch_size))

  def input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    dataset = dataset.batch(test_batch_size, drop_remainder=True)
    return dataset

  num_steps = num_test_examples // test_batch_size
  metrics = ipu_estimator.evaluate(input_fn=input_fn, steps=num_steps)
  test_loss = metrics["loss"]
  test_accuracy = metrics["accuracy"]

  print("Test loss: {:g}".format(test_loss))
  print("Test accuracy: {:.2f}%".format(100 * test_accuracy))


def main():
  args = parse_args()
  train_data, test_data = cifar10.load_data()

  num_test_examples = len(test_data[0])
  if num_test_examples % args.iterations_per_loop != 0:
    raise ValueError(
        ("iterations_per_loop ({}) must evenly divide the number of test "
         "examples ({})").format(args.iterations_per_loop, num_test_examples))

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
