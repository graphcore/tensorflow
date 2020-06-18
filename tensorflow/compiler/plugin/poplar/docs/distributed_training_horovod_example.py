import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python import ipu
from tensorflow.python.ipu import horovod as hvd
from tensorflow.python.ipu.horovod import ipu_horovod_strategy

BATCH_SIZE = 64


def input_fn(mode):  # pylint: disable=unused-argument
  train_data, _ = tf.keras.datasets.mnist.load_data()

  def normalise(image, label):
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=-1)
    label = label.astype(np.int32)
    return image, label

  x_train, y_train = normalise(*train_data)

  def generator():
    return zip(x_train, y_train)

  types = (x_train.dtype, y_train.dtype)
  shapes = (x_train.shape[1:], y_train.shape[1:])
  mnist_dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  mnist_dataset = mnist_dataset.shard(hvd.size(), hvd.rank())
  mnist_dataset = mnist_dataset.shuffle(len(y_train)) \
      .cache().batch(BATCH_SIZE, drop_remainder=True).repeat()
  return mnist_dataset


def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation="relu"),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation="relu"),
      tf.keras.layers.Dense(10)
  ])
  logits = model(features, training=mode == tf.estimator.ModeKeys.TRAIN)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {"logits": logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.AdamOptimizer()
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.compat.v1.losses.Reduction.NONE)(labels,
                                                                      logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)

  variables = model.trainable_variables

  def host_model_fn(*host_gradients):
    # This will allreduce the gradients and update the weights on the host.
    return optimizer.apply_gradients(zip(host_gradients, variables))

  train_op = tf.identity(loss)
  grads_and_vars = optimizer.compute_gradients(loss, var_list=variables)
  gradients = [g for (g, _) in grads_and_vars]
  host_call = (host_model_fn, gradients)

  return ipu.ipu_estimator.IPUEstimatorSpec(mode=mode,
                                            loss=loss,
                                            train_op=train_op,
                                            host_call=host_call)


# Initialise the Horovod runtime.
hvd.init()

# Create a Horovod strategy that places variables on the host.
strategy = ipu_horovod_strategy.IPUHorovodStrategy(variables_on_host=True)

ipu_options = ipu.utils.create_ipu_config()
ipu.utils.auto_select_ipus(ipu_options, num_ipus=1)
ipu_run_config = ipu.ipu_run_config.IPURunConfig(ipu_options=ipu_options)

config = ipu.ipu_run_config.RunConfig(
    session_config=tf.ConfigProto(allow_soft_placement=False),
    ipu_run_config=ipu_run_config,
    train_distribute=strategy,
)

parser = argparse.ArgumentParser()
parser.add_argument("--num-steps", type=int, default=10000)
parser.add_argument("--model-dir")
args = parser.parse_args()

classifier = ipu.ipu_estimator.IPUEstimator(
    config=config,
    model_fn=model_fn,
    model_dir=args.model_dir,
)

# Training progress is logged as INFO, so enable that logging level.
tf.logging.set_verbosity(tf.logging.INFO)
classifier.train(input_fn=input_fn, max_steps=args.num_steps)
