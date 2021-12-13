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
from absl.testing import parameterized
import pva

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ipu import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.training import adam
from tensorflow.python.training import rmsprop
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import loops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


def _gradient_accumulation_loop(test_wrapper,
                                fwd_fn,
                                inputs_fn,
                                input_values,
                                repeat_count,
                                num_batches_to_accumulate,
                                dataset_fn,
                                optimizer_fn,
                                num_iterations=None,
                                replication_factor=1,
                                minimum_remote_tensor_size=128,
                                replicated_optimizer_state_sharding=False,
                                assert_compute_sets_contain_list=None,
                                reduction_method=None):
  g = ops.Graph()

  if num_iterations is None:
    num_iterations = repeat_count * num_batches_to_accumulate

  with g.as_default(), test_wrapper.test_session(graph=g) as session:
    dataset = dataset_fn()
    inputs = inputs_fn()
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    with variable_scope.variable_scope("ipu", use_resource=True, reuse=False):

      def model(*args):
        loss = fwd_fn(*functional_ops._convert_to_list(args))  # pylint: disable=W0212
        enqueue_op = outfeed_queue.enqueue(loss)
        optimizer = optimizer_fn()
        if replication_factor > 1:
          opt = ga.CrossReplicaGradientAccumulationOptimizerV2(  # pylint: disable=line-too-long
              optimizer,
              num_batches_to_accumulate,
              reduction_method=reduction_method,
              offload_weight_update_variables=replicated_optimizer_state_sharding or None,  # pylint: disable=line-too-long
              replicated_optimizer_state_sharding=replicated_optimizer_state_sharding)  # pylint: disable=line-too-long
        else:
          opt = ga.GradientAccumulationOptimizerV2(
              optimizer,
              num_batches_to_accumulate,
              False,
              reduction_method=reduction_method)
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

    cfg = IPUConfig()
    if assert_compute_sets_contain_list is not None:
      report_helper = tu.ReportHelper()
      report_helper.set_autoreport_options(cfg)
    if utils.running_on_ipu_model():
      tu.enable_ipu_events(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.optimizations.minimum_remote_tensor_size = minimum_remote_tensor_size
    cfg.auto_select_ipus = replication_factor
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()
    utils.move_variable_initialization_to_cpu()

    session.run(variables.global_variables_initializer())
    session.run(infeed_queue.initializer)
    if assert_compute_sets_contain_list is not None:
      report_helper.clear_reports()
    session.run(loop_ret, feed_dict=dict(zip(inputs, input_values)))
    r = session.run(outfeed_op)
    if assert_compute_sets_contain_list is not None:
      report = pva.openReport(report_helper.find_report())
      test_wrapper.assert_compute_sets_contain_list(
          report, assert_compute_sets_contain_list)
    return r


def _compare_to_cpu(test_wrapper,
                    fwd_fn,
                    inputs_fn,
                    input_values,
                    repeat_count,
                    num_batches_to_accumulate,
                    dataset_fn,
                    optimizer_fn,
                    replication_factor=1,
                    minimum_remote_tensor_size=128,
                    replicated_optimizer_state_sharding=False,
                    assert_compute_sets_contain_list=None,
                    reduction_method=None):

  ga_losses = _gradient_accumulation_loop(
      test_wrapper,
      fwd_fn,
      inputs_fn,
      input_values,
      repeat_count,
      num_batches_to_accumulate,
      dataset_fn,
      optimizer_fn,
      replication_factor=replication_factor,
      minimum_remote_tensor_size=minimum_remote_tensor_size,
      replicated_optimizer_state_sharding=replicated_optimizer_state_sharding,
      assert_compute_sets_contain_list=assert_compute_sets_contain_list,
      reduction_method=reduction_method)

  cpu_losses = pipelining_test_util.PipelineTester._cpu_with_grad_accum(  # pylint: disable=protected-access
      test_wrapper, [fwd_fn],
      inputs_fn,
      input_values,
      repeat_count,
      num_batches_to_accumulate * replication_factor,
      dataset_fn,
      optimizer_fn,
      reduction_method=reduction_method)

  cpu_losses = np.reshape(cpu_losses, np.shape(ga_losses))
  test_wrapper.assertAllClose(cpu_losses, ga_losses)


class GradientAccumulationTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):
  @tu.skip_on_hw
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

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.9)

    with self.assertRaisesRegex(
        errors.FailedPreconditionError,
        'Detected a gradient accumulation operation with 32'):
      _gradient_accumulation_loop(
          self,
          model,
          inputs_fn, [10.01],
          3,
          32,
          dataset_fn,
          optimizer_fn,
          10,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)  # pylint: disable=line-too-long

  @tu.test_may_use_ipus_or_model(num_ipus=1)
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

    def optimizer_fn():
      return adam.AdamOptimizer()

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

    _compare_to_cpu(self,
                    fwd_fn,
                    inputs_fn, [10.01],
                    repeat_count,
                    num_batches_to_accumulate,
                    dataset_fn,
                    optimizer_fn,
                    reduction_method=ga.GradientAccumulationReductionMethod.MEAN)  # pylint: disable=line-too-long

  @tu.test_may_use_ipus_or_model(num_ipus=1)
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

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

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

    _compare_to_cpu(self,
                    fwd_fn,
                    lambda: [], [],
                    repeat_count,
                    num_batches_to_accumulate,
                    dataset_fn,
                    optimizer_fn,
                    reduction_method=ga.GradientAccumulationReductionMethod.MEAN)  # pylint: disable=line-too-long

  @tu.test_may_use_ipus_or_model(num_ipus=1)
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

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.8)

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

    _compare_to_cpu(self,
                    fwd_fn,
                    lambda: [], [],
                    repeat_count,
                    num_batches_to_accumulate,
                    dataset_fn,
                    optimizer_fn,
                    reduction_method=ga.GradientAccumulationReductionMethod.MEAN)  # pylint: disable=line-too-long

  @tu.test_may_use_ipus_or_model(num_ipus=1)
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

    def optimizer_fn():
      return adam.AdamOptimizer()

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

    _compare_to_cpu(self,
                    fwd_fn,
                    lambda: [], [],
                    repeat_count,
                    num_batches_to_accumulate,
                    dataset_fn,
                    optimizer_fn,
                    reduction_method=ga.GradientAccumulationReductionMethod.MEAN)  # pylint: disable=line-too-long

  @tu.test_may_use_ipus_or_model(num_ipus=1)
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

    def optimizer_fn():
      return adam.AdamOptimizer()

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

    _compare_to_cpu(self,
                    fwd_fn,
                    lambda: [], [],
                    repeat_count,
                    num_batches_to_accumulate,
                    dataset_fn,
                    optimizer_fn,
                    reduction_method=ga.GradientAccumulationReductionMethod.MEAN)  # pylint: disable=line-too-long

  def _compare6(self, optimizer_fn, replicated_optimizer_state_sharding=False):
    dataset_size = 10
    num_batches_to_accumulate = 4
    repeat_count = 2
    embedding_size = 4

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

    _compare_to_cpu(
        self,
        fwd_fn,
        lambda: [], [],
        repeat_count,
        num_batches_to_accumulate,
        dataset_fn,
        optimizer_fn,
        replication_factor=2,
        minimum_remote_tensor_size=8,
        replicated_optimizer_state_sharding=replicated_optimizer_state_sharding,
        assert_compute_sets_contain_list=['/ReduceScatter/']
        if replicated_optimizer_state_sharding else None,
        reduction_method=ga.GradientAccumulationReductionMethod.MEAN)  # pylint: disable=line-too-long

  @parameterized.parameters({'replicated_optimizer_state_sharding': False}, {
      'replicated_optimizer_state_sharding': True,
  })
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCompare6Momentum(self, replicated_optimizer_state_sharding):
    self._compare6(lambda: momentum.MomentumOptimizer(0.01, 0.8),
                   replicated_optimizer_state_sharding)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCompare6SDG(self):
    self._compare6(lambda: gradient_descent.GradientDescentOptimizer(0.01))

  @parameterized.parameters({'replicated_optimizer_state_sharding': False}, {
      'replicated_optimizer_state_sharding': True,
  })
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCompare6Adam(self, replicated_optimizer_state_sharding):
    self._compare6(adam.AdamOptimizer, replicated_optimizer_state_sharding)

  @parameterized.parameters({'replicated_optimizer_state_sharding': False}, {
      'replicated_optimizer_state_sharding': True,
  })
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testCompare6RMS(self, replicated_optimizer_state_sharding):
    self._compare6(lambda: rmsprop.RMSPropOptimizer(0.01),
                   replicated_optimizer_state_sharding)

  @tu.test_may_use_ipus_or_model(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testGradientAccumulationDtype(self):
    gradient_accumulation_count = 8
    gradient_accumulation_dtype = np.float32

    x = np.finfo(np.float16).max
    y = np.array(0.0, dtype=np.float16)
    initial_w = np.array(1.0, dtype=np.float16)
    learning_rate = 2**-10

    features = np.repeat(x, gradient_accumulation_count)
    labels = np.repeat(y, gradient_accumulation_count)
    dataset = dataset_ops.Dataset.from_tensor_slices((features, labels))

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    grad_outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    class CastingGradientDescent(optimizer_lib.Optimizer):  # pylint: disable=abstract-method
      """Compute update using the dtype of the gradient, and then cast to
      the dtype of the variable."""
      def __init__(self, outer):
        self.outer = outer
        super().__init__(use_locking=False, name="CastingGradientDescent")

      def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        update_ops = []

        for (grad, var) in grads_and_vars:
          self.outer.assertEqual(grad.dtype, gradient_accumulation_dtype)
          update_ops.append(grad_outfeed_queue.enqueue(grad))
          delta = math_ops.cast(-learning_rate * grad, var.dtype)
          update_ops.append(var.assign_add(delta))

        return control_flow_ops.group(*update_ops)

    def model_iteration(features, labels):
      w = variable_scope.get_variable(name="w", initializer=initial_w)
      partial = w * features
      loss = partial + labels

      def dtype_getter(var):
        self.assertEqual(var, w)
        return gradient_accumulation_dtype

      opt = ga.GradientAccumulationOptimizerV2(
          CastingGradientDescent(self),
          gradient_accumulation_count,
          dtype=dtype_getter,
          reduction_method=ga.GradientAccumulationReductionMethod.SUM)  # pylint: disable=line-too-long
      return opt.minimize(loss)

    def model():
      return loops.repeat(gradient_accumulation_count,
                          model_iteration,
                          infeed_queue=infeed_queue)

    def compiled_model():
      with ops.device("/device:IPU:0"):
        return ipu_compiler.compile(model)

    train_op = compiled_model()

    dequeued_gradient = grad_outfeed_queue.dequeue()

    cfg = IPUConfig()
    if utils.running_on_ipu_model():
      tu.enable_ipu_events(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 1
    tu.add_hw_ci_connection_options(cfg)
    cfg.configure_ipu_system()
    utils.move_variable_initialization_to_cpu()

    with tu.ipu_session() as sess:
      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())

      sess.run(train_op)
      [actual_accumulated_gradient] = sess.run(dequeued_gradient)

      # L(x) = w * x + y
      # dL(x)/dw = x
      # This would overflow in fp16:
      expected_accumulated_gradient = gradient_accumulation_count * x.astype(
          gradient_accumulation_dtype)

      self.assertAllEqual(expected_accumulated_gradient,
                          actual_accumulated_gradient)

      sess.run(infeed_queue.deleter)
      sess.run(grad_outfeed_queue.deleter)

  def __makeGATestNetwork(self, reduction_method):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    num_batches_to_accumulate = 20
    repeat_count = 4
    optimizer = adam.AdamOptimizer(learning_rate=1.0,
                                   epsilon=1.0,
                                   beta1=0.5,
                                   beta2=0.5)

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

    g = ops.Graph()

    repeat_count = 2
    num_batches_to_accumulate = 4

    num_iterations = repeat_count * num_batches_to_accumulate

    with g.as_default(), self.test_session(graph=g):
      dataset = dataset_fn()
      inputs = inputs_fn()
      infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      with variable_scope.variable_scope("ipu", use_resource=True,
                                         reuse=False):

        def model(*args):
          loss = fwd_fn(*functional_ops._convert_to_list(args))  # pylint: disable=W0212,E1120
          enqueue_op = outfeed_queue.enqueue(loss)
          opt = ga.GradientAccumulationOptimizerV2(
              optimizer,
              num_batches_to_accumulate,
              reduction_method=reduction_method)
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
        ipu_compiler.compile(my_net, inputs=inputs)

  @test_util.deprecated_graph_mode_only
  def testGAReduceMethodNone(self):
    with self.assertRaisesRegex(
        ValueError, 'reduction_method must be set to SUM, MEAN or '
        'RUNNING_MEAN'):
      self.__makeGATestNetwork(None)

  @parameterized.parameters([
      'SUM', 'sum', 'MEAN', 'mean', ga.GradientAccumulationReductionMethod.SUM,
      ga.GradientAccumulationReductionMethod.MEAN
  ])
  @test_util.deprecated_graph_mode_only
  def testGAReduceMethodSupported(self, reduction_method):
    with ops.device("/device:IPU:0"):
      self.__makeGATestNetwork(reduction_method)

  @parameterized.parameters([
      'RUNNING_MEAN', 'running_mean',
      ga.GradientAccumulationReductionMethod.RUNNING_MEAN
  ])
  @test_util.deprecated_graph_mode_only
  def testGAReduceMethodUnsupported(self, reduction_method):
    with self.assertRaisesRegex(
        ValueError, 'Only GradientAccumulationReductionMethod.SUM and '
        'GradientAccumulationReductionMethod.MEAN are '
        'supported at the moment'):
      self.__makeGATestNetwork(reduction_method)

  @parameterized.parameters(['Exp', 10])
  @test_util.deprecated_graph_mode_only
  def testGAReduceMethodInvalid(self, reduction_method):
    with self.assertRaisesRegex(
        ValueError, 'reduction_method must be set to SUM, MEAN '
        'or RUNNING_MEAN'):
      self.__makeGATestNetwork(reduction_method)


if __name__ == "__main__":
  googletest.main()
