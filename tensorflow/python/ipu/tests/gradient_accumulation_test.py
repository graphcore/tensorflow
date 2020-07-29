# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import numpy as np

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training import adam
from tensorflow.python.training import rmsprop
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


def next_feed_id():
  result = 'feed' + str(next_feed_id.feed_count)
  next_feed_id.feed_count += 1
  return result


next_feed_id.feed_count = 0


def _gradient_accumulation_loop(test_wrapper,
                                fwd_fn,
                                inputs_fn,
                                input_values,
                                repeat_count,
                                num_batches_to_accumulate,
                                dataset_fn,
                                optimizer,
                                num_iterations=None):
  g = ops.Graph()

  if num_iterations is None:
    num_iterations = repeat_count * num_batches_to_accumulate

  with g.as_default(), test_wrapper.test_session(graph=g) as session:
    dataset = dataset_fn()
    inputs = inputs_fn()
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, next_feed_id())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(next_feed_id())

    with variable_scope.variable_scope("ipu", use_resource=True, reuse=False):

      def model(*args):
        loss = fwd_fn(*functional_ops._convert_to_list(args))  # pylint: disable=W0212
        enqueue_op = outfeed_queue.enqueue(loss)
        opt = gradient_accumulation_optimizer.GradientAccumulationOptimizerV2(
            optimizer, num_batches_to_accumulate)
        outs = list(args[:len(args) - infeed_queue.number_of_tuple_elements])
        outs.append(enqueue_op)
        outs.append(opt.minimize(loss))
        return outs

      def my_net(*args):
        return loops.repeat(num_iterations,
                            model,
                            inputs=args,
                            infeed_queue=infeed_queue)

    with ops.device("/device:IPU:0"):
      loop_ret = ipu_compiler.compile(my_net, inputs=inputs)

    outfeed_op = outfeed_queue.dequeue()

    profiling = utils.running_on_ipu_model()

    cfg = utils.create_ipu_config(profiling=profiling,
                                  profile_execution=profiling)
    cfg = utils.auto_select_ipus(cfg, 1)
    utils.configure_ipu_system(cfg)
    utils.move_variable_initialization_to_cpu()

    session.run(variables.global_variables_initializer())
    session.run(infeed_queue.initializer)
    session.run(loop_ret, feed_dict=dict(zip(inputs, input_values)))
    return session.run(outfeed_op)


def _compare_to_cpu(test_wrapper, fwd_fn, inputs_fn, input_values,
                    repeat_count, num_batches_to_accumulate, dataset_fn,
                    optimizer):

  ga_losses = _gradient_accumulation_loop(test_wrapper, fwd_fn, inputs_fn,
                                          input_values, repeat_count,
                                          num_batches_to_accumulate,
                                          dataset_fn, optimizer)

  cpu_losses = pipelining_test_util.PipelineTester._cpu_with_grad_accum(  # pylint: disable=protected-access
      test_wrapper, [fwd_fn], inputs_fn, input_values, repeat_count,
      num_batches_to_accumulate, dataset_fn, optimizer)

  test_wrapper.assertAllClose(cpu_losses, ga_losses)


class GradientAccumulationTest(test_util.TensorFlowTestCase):
  @test_util.deprecated_graph_mode_only
  def testIterationsNotMultiple(self):
    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return a, b

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)
      return dataset.map(dataset_parser)

    def model(c, x, b):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(x)
      y = y + b
      y = math_ops.reduce_sum(y) + c
      return y

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        'Detected a gradient accumulation operation with 32'):
      _gradient_accumulation_loop(self, model, inputs_fn, [10.01], 3, 32,
                                  dataset_fn,
                                  momentum.MomentumOptimizer(0.01, 0.9), 10)

  @test_util.deprecated_graph_mode_only
  def testCompare1(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    num_batches_to_accumulate = 20
    repeat_count = 2
    optimizer = adam.AdamOptimizer()

    def fwd_fn(c, img, label):
      with variable_scope.variable_scope("part1", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5),
            name='conv1')(img)
      y = y * 20
      y = layers.Dense(2,
                       kernel_initializer=init_ops.constant_initializer(0.5),
                       bias_initializer=init_ops.constant_initializer(0.5))(y)
      return math_ops.reduce_sum(
          layers.Dense(2,
                       kernel_initializer=init_ops.constant_initializer(0.5),
                       bias_initializer=init_ops.constant_initializer(0.5))
          (y)) + c + label

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    _compare_to_cpu(self, fwd_fn, inputs_fn, [10.01], repeat_count,
                    num_batches_to_accumulate, dataset_fn, optimizer)

  @test_util.deprecated_graph_mode_only
  def testCompare2(self):
    # Resnet like network.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(100, shape=[4])
      dataset = dataset.batch(batch_size=64, drop_remainder=True)
      dataset = dataset.batch(batch_size=64, drop_remainder=True)
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = math_ops.reduce_mean(img, axis=[1, 2, 3])
        return img, math_ops.cast(label, np.int32)

      return dataset.map(dataset_parser)

    num_batches_to_accumulate = 18
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

    def fixed_padding(inputs, kernel_size):
      pad_total = kernel_size - 1
      pad_beg = pad_total // 2
      pad_end = pad_total - pad_beg
      padded_inputs = array_ops.pad(
          inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      return padded_inputs

    def block(name, first_stride, out_filters, count, x):

      for i in range(count):
        shape_in = x.shape
        stride = first_stride if (i == 0) else 1
        if stride > 1:
          x = fixed_padding(x, 3)
        sc = x

        with variable_scope.variable_scope(name + "/" + str(i) + "/1"):
          x = conv(x, 3, stride, out_filters)
          x = nn.relu(x)

        with variable_scope.variable_scope(name + "/" + str(i) + "/2"):
          x = conv(x, 3, 1, out_filters)

          # shortcut
          if stride != 1:
            sc = array_ops.strided_slice(sc, [0, 0, 0, 0],
                                         sc.shape,
                                         strides=[1, stride, stride, 1])
          pad = int(x.shape[3] - shape_in[3])
          if pad != 0:
            sc = array_ops.pad(sc, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]])

          x = nn.relu(x + sc)

      return x

    def fc(x, num_units_out):
      return layers.Dense(
          num_units_out,
          kernel_initializer=init_ops.constant_initializer(0.01),
          bias_initializer=init_ops.constant_initializer(0.05))(x)

    def max_pool(x, ksize=3, stride=2):
      return layers.MaxPooling2D(ksize, stride, padding='SAME')(x)

    def conv(x, ksize, stride, filters_out):
      return layers.Conv2D(
          filters_out,
          ksize,
          stride,
          'SAME',
          kernel_initializer=init_ops.constant_initializer(0.01),
          bias_initializer=init_ops.constant_initializer(0.05))(x)

    def fwd_fn(img, label):
      with variable_scope.variable_scope("part1", use_resource=True):
        x = conv(img, 7, 2, 8)
        x = nn.relu(x)
        x = max_pool(x, ksize=3, stride=2)

      with variable_scope.variable_scope("part2", use_resource=True):
        x = block("b", 2, 32, 1, x)

      with variable_scope.variable_scope("part3", use_resource=True):
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = fc(x, 100)
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=label))
        return loss

    _compare_to_cpu(self, fwd_fn, lambda: [], [], repeat_count,
                    num_batches_to_accumulate, dataset_fn, optimizer)

  @test_util.deprecated_graph_mode_only
  def testCompare3(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int32), math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    num_batches_to_accumulate = 20
    repeat_count = 2
    optimizer = momentum.MomentumOptimizer(0.01, 0.8)

    def fwd_fn(idx, label):
      with variable_scope.variable_scope("part1", use_resource=True):
        embedding = variable_scope.get_variable(
            "c",
            shape=[10, 1216],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(10.01),
            trainable=True)
      x = embedding_ops.embedding_lookup(embedding, idx)

      logits = math_ops.reduce_sum(x, axis=[-1])
      loss = math_ops.reduce_mean(
          nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=label))
      return loss

    _compare_to_cpu(self, fwd_fn, lambda: [], [], repeat_count,
                    num_batches_to_accumulate, dataset_fn, optimizer)

  @test_util.deprecated_graph_mode_only
  def testCompare4(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    num_batches_to_accumulate = 20
    repeat_count = 2
    optimizer = adam.AdamOptimizer()

    def fwd_fn(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)

      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w1",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)

      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)

      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w3",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)

      # Ruse the weight here.
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      logits = math_ops.reduce_mean(x, axis=[1])
      loss = math_ops.reduce_mean(
          nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=label))
      return loss

    _compare_to_cpu(self, fwd_fn, lambda: [], [], repeat_count,
                    num_batches_to_accumulate, dataset_fn, optimizer)

  @test_util.deprecated_graph_mode_only
  def testCompare5(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    num_batches_to_accumulate = 20
    repeat_count = 2
    optimizer = adam.AdamOptimizer()

    def fwd_fn(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      x = nn.relu(x)

      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      logits = math_ops.reduce_mean(x, axis=[1])
      loss = math_ops.reduce_mean(
          nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=label))
      return loss

    _compare_to_cpu(self, fwd_fn, lambda: [], [], repeat_count,
                    num_batches_to_accumulate, dataset_fn, optimizer)

  def _compare6(self, optimizer):
    dataset_size = 100
    num_batches_to_accumulate = 10
    repeat_count = 2
    embedding_size = 10

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(dataset_size, shape=[4])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int32), math_ops.cast(label % 4, np.int32)

      return dataset.map(dataset_parser)

    def fwd_fn(idx, label):
      np.random.seed(1)
      embedding_shape = (dataset_size, embedding_size)
      embedding_initializer = np.random.normal(0, 1, embedding_shape).astype(
          np.float32)
      weights_shape = (embedding_size, embedding_size)
      weights_initializer = np.random.normal(0, 1,
                                             weights_shape).astype(np.float32)

      with variable_scope.variable_scope("part1", use_resource=True):
        embedding = variable_scope.get_variable(
            "c",
            dtype=np.float32,
            initializer=embedding_initializer,
            trainable=True)

        weight = variable_scope.get_variable("w0",
                                             dtype=np.float32,
                                             initializer=weights_initializer,
                                             trainable=True)

      x = embedding_ops.embedding_lookup(embedding, idx)
      x = math_ops.matmul(x, weight)

      logits = math_ops.reduce_sum(x, axis=[-1])
      loss = math_ops.reduce_mean(
          nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=label))
      return loss

    _compare_to_cpu(self, fwd_fn, lambda: [], [], repeat_count,
                    num_batches_to_accumulate, dataset_fn, optimizer)

  @test_util.deprecated_graph_mode_only
  def testCompare6Momentum(self):
    self._compare6(momentum.MomentumOptimizer(0.01, 0.8))

  @test_util.deprecated_graph_mode_only
  def testCompare6SDG(self):
    self._compare6(gradient_descent.GradientDescentOptimizer(0.01))

  @test_util.deprecated_graph_mode_only
  def testCompare6Adam(self):
    self._compare6(adam.AdamOptimizer())

  @test_util.deprecated_graph_mode_only
  def testCompare6RMS(self):
    self._compare6(rmsprop.RMSPropOptimizer(0.01))


if __name__ == "__main__":
  googletest.main()
