# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import os
import tempfile
import subprocess
from functools import partial
from absl.testing import parameterized
import numpy as np

from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import metrics
from tensorflow.python.ops import nn, nn_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.framework import ops, dtypes
from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
from tensorflow.python.ipu import utils, ipu_compiler, scopes, loops, ipu_infeed_queue, ipu_outfeed_queue
from tensorflow.python.ipu import dataset_benchmark
from tensorflow.python.ipu import rand_ops
from tensorflow.compat.v1 import disable_v2_behavior
from tensorflow.python.training import momentum
from tensorflow.python.keras.datasets import mnist
from tensorflow.compat.v1 import train

ops.disable_eager_execution()
disable_v2_behavior()

L1_SIZE = 320
L2_SIZE = 10

tmp_dir_obj = tempfile.TemporaryDirectory()
tmp_dir = tmp_dir_obj.name


def dense_layer(hiddenSize, input_, scope_name):
  with variable_scope.variable_scope(scope_name,
                                     reuse=variable_scope.AUTO_REUSE,
                                     use_resource=True):
    w = variable_scope.get_variable(
        "weight",
        shape=[input_.shape[-1], hiddenSize],
        initializer=init_ops.glorot_uniform_initializer())
    b = variable_scope.get_variable("bias",
                                    shape=[hiddenSize],
                                    initializer=init_ops.zeros_initializer())
    return nn.relu_layer(input_, w, b)


def train_model(lr, outqueue, inputs, labels):
  h1Size = L1_SIZE
  h2Size = L2_SIZE
  droprate = 0.2

  relu1 = dense_layer(h1Size, inputs, "d1")
  drop1 = rand_ops.dropout(relu1, rate=droprate)
  relu2 = dense_layer(h2Size, drop1, "d2")

  with variable_scope.variable_scope("metrics",
                                     reuse=variable_scope.AUTO_REUSE,
                                     use_resource=True):
    acc, acc_op = metrics.accuracy(
        labels=labels,
        predictions=math_ops.argmax(relu2, axis=1, output_type=dtypes.int32),
        name="accuracy")
    loss = nn_ops.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=relu2)

  with variable_scope.variable_scope("training",
                                     reuse=variable_scope.AUTO_REUSE,
                                     use_resource=True):
    optimiser = momentum.MomentumOptimizer(learning_rate=lr,
                                           momentum=0.0001,
                                           use_nesterov=True,
                                           name='optimise')
    train_op = optimiser.minimize(loss)
    with ops.control_dependencies([train_op, acc_op]):
      mean_loss = math_ops.reduce_mean(loss, name='train_loss')

  return outqueue.enqueue({'mean_loss': mean_loss, 'acc': acc})


def test_model(outqueue, inputs):
  h1Size = L1_SIZE
  h2Size = L2_SIZE

  relu1 = dense_layer(h1Size, inputs, "d1")
  drop1 = relu1
  relu2 = dense_layer(h2Size, drop1, "d2")
  predictions = math_ops.argmax(relu2, axis=1, output_type=dtypes.int32)

  return outqueue.enqueue({'predictions': predictions})


def scheduler(epoch):
  if epoch < 1:
    return 0.02
  if epoch < 3:
    return 0.01
  return 0.001


def loop_builder(iterations, builder_func, infeed):
  return loops.repeat(iterations, builder_func, [], infeed)


def run_model():
  # Use Keras to get the dataset:
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  n_train = 240
  n_test = 48

  x_train = x_train[0:n_train, :, :]
  y_train = y_train[0:n_train]
  x_test = x_test[0:n_test, :, :]
  y_test = y_test[0:n_test]

  # Sizes/shapes for the dataset:
  image_shape = x_train.shape[1:]
  num_pixels = image_shape[0] * image_shape[1]
  batch_size = 16
  num_train = y_train.shape[0]
  num_test = y_test.shape[0]
  data_shape = [None, num_pixels]

  # Flatten the images and cast the labels:
  x_train_flat = x_train.astype(np.float32).reshape(-1, num_pixels)
  x_test_flat = x_test.astype(np.float32).reshape(-1, num_pixels)
  y_train = y_train.astype(np.int32)
  y_test = y_test.astype(np.int32)

  # Decide how to split epochs into loops up front:
  epochs = 5
  ipu_steps_per_epoch = 15
  batches_per_epoch = num_train // batch_size
  test_batches = num_test // batch_size
  batches_per_step = batches_per_epoch // ipu_steps_per_epoch
  if not batches_per_epoch % ipu_steps_per_epoch == 0:
    raise ValueError(f"IPU steps per epoch {ipu_steps_per_epoch} " +
                     f"must divide batches per epoch {batches_per_epoch}.")

  # Put placeholders on the CPU host:
  with ops.device("cpu"):
    place_x = array_ops.placeholder(dtype=dtypes.float32,
                                    shape=data_shape,
                                    name="input")
    place_y = array_ops.placeholder(dtype=dtypes.int32,
                                    shape=[None],
                                    name="label")
    lr_placeholder = array_ops.placeholder(dtypes.float32, shape=[])

  # Create dataset and IPU feeds:
  train_dataset = dataset_ops.Dataset.from_tensor_slices((place_x, place_y))
  train_dataset = train_dataset.cache().repeat().batch(batch_size,
                                                       drop_remainder=True)

  test_dataset = dataset_ops.Dataset.from_tensor_slices((place_x,))
  test_dataset = test_dataset.cache().repeat().batch(batch_size,
                                                     drop_remainder=True)

  infeed_train_queue = ipu_infeed_queue.IPUInfeedQueue(
      train_dataset, feed_name="train_infeed")
  outfeed_train_queue = ipu_outfeed_queue.IPUOutfeedQueue(
      feed_name="train_outfeed")
  infeed_test_queue = ipu_infeed_queue.IPUInfeedQueue(test_dataset,
                                                      feed_name="test_infeed")
  outfeed_test_queue = ipu_outfeed_queue.IPUOutfeedQueue(
      feed_name="test_outfeed")

  # Use function binding to create all the builder functions that are neeeded:
  bound_train_model = partial(train_model, lr_placeholder, outfeed_train_queue)
  bound_train_loop = partial(loop_builder, batches_per_step, bound_train_model,
                             infeed_train_queue)
  bound_test_model = partial(test_model, outfeed_test_queue)
  bound_test_loop = partial(loop_builder, test_batches, bound_test_model,
                            infeed_test_queue)

  # Use the bound builder functions to place the model on the IPU:
  with scopes.ipu_scope("/device:IPU:0"):
    train_loop = ipu_compiler.compile(bound_train_loop, inputs=[])
    test_loop = ipu_compiler.compile(bound_test_loop, inputs=[])

  # Initialisers should go on the CPU:
  with ops.device("cpu"):
    metrics_vars = ops.get_collection(ops.GraphKeys.LOCAL_VARIABLES,
                                      scope="metrics")
    metrics_initializer = variables.variables_initializer(
        var_list=metrics_vars)
    saver = train.Saver()

  # Setup and acquire an IPU device:
  config = utils.create_ipu_config()
  config = utils.auto_select_ipus(config, 1)
  utils.configure_ipu_system(config)

  # These allow us to retrieve the results of IPU feeds:
  dequeue_train_outfeed = outfeed_train_queue.dequeue()
  dequeue_test_outfeed = outfeed_test_queue.dequeue()

  # Create a benchmark program for the infeed to determine maximum achievable throughput:
  infeed_perf = dataset_benchmark.infeed_benchmark(infeed_train_queue, epochs,
                                                   num_train, True)

  # Run the model:
  with sl.Session() as sess:
    print(f"  Benchmarking the infeed...")
    sess.run(infeed_perf, feed_dict={place_x: x_train_flat, place_y: y_train})

    sess.run(variables.global_variables_initializer())
    sess.run(infeed_train_queue.initializer,
             feed_dict={
                 place_x: x_train_flat,
                 place_y: y_train
             })

    print(f"  Training...")
    for e in range(epochs):
      sess.run(metrics_initializer)
      for _ in range(ipu_steps_per_epoch):
        sess.run(train_loop, feed_dict={lr_placeholder: scheduler(e)})
        result = sess.run(dequeue_train_outfeed)

    model_save_path = f'{tmp_dir}/model'
    saver.save(sess, model_save_path)

    print(f"  Testing...")

    sess.run(metrics_initializer)
    sess.run(infeed_test_queue.initializer, feed_dict={place_x: x_test_flat})
    sess.run(test_loop)
    result = sess.run(dequeue_test_outfeed)

    d1_bias = train.load_variable(model_save_path, 'd1/bias')
    d1_weight = train.load_variable(model_save_path, 'd1/weight')
    d2_bias = train.load_variable(model_save_path, 'd2/bias')
    d2_weight = train.load_variable(model_save_path, 'd2/weight')

    mnist_ref = dict(d1_bias=d1_bias,
                     d1_weight=d1_weight,
                     d2_bias=d2_bias,
                     d2_weight=d2_weight,
                     images=x_test_flat,
                     labels=result['predictions'])

    return mnist_ref


class ApplicationRuntimeTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):
  @test_util.deprecated_graph_mode_only
  def testApplicationRuntime(self):
    poplar_flags = os.environ.get('TF_POPLAR_FLAGS', '')
    poplar_flags += f' --executable_cache_path={tmp_dir}'

    with test.mock.patch.dict("os.environ", {"TF_POPLAR_FLAGS": poplar_flags}):
      print('Running MNIST example')
      mnist_ref = run_model()
      print('Example done')

    ret = subprocess.run(['/bin/ls', '-At', f'{tmp_dir}/'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         encoding='utf-8')

    poplar_exec_filepath = f'{tmp_dir}/{ret.stdout.strip().split()[0]}'

    print('Running MNIST through embedded runtime')

    input_descs = [
        ('XLA_Args/d1/bias', [L1_SIZE], dtypes.float32),  #0
        ('XLA_Args/d1/weight', [784, L1_SIZE], dtypes.float32),  #1
        ('XLA_Args/d2/bias', [L2_SIZE], dtypes.float32),  #2
        ('XLA_Args/d2/weight', [L1_SIZE, L2_SIZE], dtypes.float32),  #3
        ('XLA_Args/metrics_1/accuracy/count', [], dtypes.float32),  #4
        ('XLA_Args/metrics_1/accuracy/total', [], dtypes.float32),  #5
    ]

    input_placeholders = []
    for name, shape, dtype in input_descs:
      input_ph = array_ops.placeholder(dtype, shape=shape, name=name)
      input_placeholders.append(input_ph)

    inputs = [None] * 6
    inputs[0] = mnist_ref['d1_bias']
    inputs[1] = mnist_ref['d1_weight']
    inputs[2] = mnist_ref['d2_bias']
    inputs[3] = mnist_ref['d2_weight']
    inputs[4] = np.float32(0)
    inputs[5] = np.float32(0)

    inputs = tuple(inputs)
    input_placeholders = tuple(input_placeholders)

    images = array_ops.placeholder(dtypes.float32,
                                   shape=[16, 784],
                                   name='images')

    images_all = mnist_ref['images']
    labels_all = np.empty(mnist_ref['labels'].shape, dtype='int32')

    with sl.Session() as session:
      for i in range(1):
        engine_name = f'mnist_engine_{i}'

        run_app = gen_application_runtime.application_runtime(
            inputs=inputs,
            filename=poplar_exec_filepath,
            engine_name=engine_name)

        for j in range(3):
          infeeds = (images,)
          with ops.control_dependencies([run_app]):
            result = gen_application_runtime.application_call(
                infeeds, outfeed_types=[dtypes.int32], engine_name=engine_name)

          images_host = images_all[j * 16:(j + 1) * 16, :]

          session.run(variables.global_variables_initializer())
          results = session.run(result,
                                feed_dict={
                                    infeeds: (images_host,),
                                    input_placeholders: inputs,
                                })

          labels_all[j, :] = results[0]

    self.assertAllClose(mnist_ref['labels'], labels_all)


if __name__ == "__main__":
  googletest.main()
