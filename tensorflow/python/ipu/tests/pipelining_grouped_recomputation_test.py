# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
import pva
import numpy as np
from absl.testing import parameterized

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.ipu import custom_ops
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import internal_ops
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import rand_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


class PipeliningGroupedRecomputationTest(test_util.TensorFlowTestCase,
                                         parameterized.TestCase):
  @parameterized.parameters([0, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare1(self, number_of_io_tiles):
    if utils.running_on_ipu_model():
      self.skipTest("Replicated top level graphs are not supported on the "
                    "IPU_MODEL target")

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 16
    repeat_count = 1

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(c, img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5),
            name='conv1')(img)
        return y, c, label

    def stage2(x, c, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        return x * 20, c, label

    def stage3(x, c, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return layers.Dense(
            2,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5))(x), c, label

    def stage4(x, c, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        return math_ops.reduce_sum(
            layers.Dense(2,
                         kernel_initializer=init_ops.constant_initializer(0.5),
                         bias_initializer=init_ops.constant_initializer(0.5))
            (x)) + c + label

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        14770,
        True,
        pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare2(self, number_of_io_tiles):
    # Resnet like network.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(100, shape=[4])
      dataset = dataset.batch(batch_size=32, drop_remainder=True)
      dataset = dataset.batch(batch_size=32, drop_remainder=True)
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = math_ops.reduce_mean(img, axis=[1, 2, 3])
        return img, math_ops.cast(label, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 18
    repeat_count = 1

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
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def max_pool(x, ksize=3, stride=2):
      return layers.MaxPooling2D(ksize, stride, padding='SAME')(x)

    def conv(x, ksize, stride, filters_out):
      return layers.Conv2D(
          filters_out,
          ksize,
          stride,
          'SAME',
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def stage1(img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        x = conv(img, 7, 2, 16)
        x = nn.relu(x)
        x = max_pool(x, ksize=3, stride=2)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = block("b", 2, 64, 1, x)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = fc(x, 100)
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                            labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        40549,
        True,
        pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare3(self, number_of_io_tiles):
    if utils.running_on_ipu_model():
      self.skipTest("Replicated top level graphs are not supported on the "
                    "IPU_MODEL target")

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int32), math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 16
    repeat_count = 1

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(idx, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        embedding = variable_scope.get_variable(
            "c",
            shape=[10, 1216],
            dtype=np.float32,
            initializer=init_ops.constant_initializer(10.01),
            trainable=True)
        x = embedding_ops.embedding_lookup(embedding, idx)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return x, label

    def stage4(x, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        logits = math_ops.reduce_sum(x, axis=[-1])
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        13681,
        True,
        pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare4(self, number_of_io_tiles):
    if utils.running_on_ipu_model():
      self.skipTest("Replicated top level graphs are not supported on the "
                    "IPU_MODEL target")
    # Stage3 has a stateful op there it cannot be recomputed.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 16
    repeat_count = 1

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(c, img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5),
            name='conv1')(img)
        return y, c, label

    def stage2(x, c, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        with ops.control_dependencies([internal_ops.print_tensor(x)]):
          return x * 20, c, label

    def stage3(x, c, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return layers.Dense(
            2,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5))(x), c, label

    def stage4(x, c, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        return math_ops.reduce_sum(
            layers.Dense(2,
                         kernel_initializer=init_ops.constant_initializer(0.5),
                         bias_initializer=init_ops.constant_initializer(0.5))
            (x)) + c + label

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        20354,
        True,
        pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare5(self, number_of_io_tiles):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(128, shape=[1, 1, 1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 16
    repeat_count = 1

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(c, img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5),
            name='conv1')(img)
        return y, c, label

    def stage2(x, c, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        return x, c, label

    def stage3(x, c, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return x, c, label

    def stage4(x, c, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        return math_ops.reduce_sum(x) + c + label

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        9174,
        True,
        pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare6(self, number_of_io_tiles):
    # Stage2 has a stateful op whose state will be stored and the rest of the
    # stage should be recomputed.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[1, 10])
      return dataset.repeat().batch(4, drop_remainder=True)

    gradient_accumulation_count = 12
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    cwd = os.getcwd()
    lib_path = os.path.join(
        cwd, "tensorflow/python/ipu/libpipelining_stateful_op.so")

    def stage1(x):
      with variable_scope.variable_scope("stage1", use_resource=True):
        weight = variable_scope.get_variable(
            'weight',
            shape=(x.shape[-1],),
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        activations = weight * (x + x)
        outputs = {
            "output_types": [np.float32],
            "output_shapes": [activations.shape],
        }
        activations, = custom_ops.precompiled_user_op([activations],
                                                      lib_path,
                                                      separate_gradients=True,
                                                      outs=outputs)
        return activations * 2

    def stage2(activations):
      return activations * 2

    def stage3(activations):
      return math_ops.reduce_sum(math_ops.reduce_mean(activations**2, -1))

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        12609,
        True,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare7(self, number_of_io_tiles):
    if utils.running_on_ipu_model():
      self.skipTest("Replicated top level graphs are not supported on the "
                    "IPU_MODEL target")
    # Stage 1 and 2 don't have a backward stage.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4, 2])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 16
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(c, img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        return img, c, label

    def stage2(x, c, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        with ops.control_dependencies([internal_ops.print_tensor(x)]):
          return x * 20, c, label

    def stage3(x, c, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        return layers.Dense(
            2,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5))(x), c, label

    def stage4(x, c, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        return math_ops.reduce_sum(x) + c + label

    def inputs_fn():
      with ops.device('cpu'):
        return [array_ops.placeholder(np.float32, shape=[])]

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        14502,
        True,
        pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights(self, number_of_io_tiles):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        x = array_ops.pad(x, [[0, 0], [0, 0], [1, 1]])
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("vs2", use_resource=True):
        weight = variable_scope.get_variable(
            "w1",
            shape=[6, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      return x, label

    def stage3(x, label):
      x = x + 1
      return x, label

    def stage4(x, label):
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
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=label))
        return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        recomp=True,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        recomputation_mode=pipelining_ops.RecomputationMode.
        RecomputeAndBackpropagateInterleaved,
        device_mapping=[0, 1, 2, 0],
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights2(self, number_of_io_tiles):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage2(x, label):
      internal_ops.print_tensor(x)
      return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage4(x, label):
      logits = math_ops.reduce_sum(x, axis=[1])
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        recomp=True,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        device_mapping=[0, 1, 0, 2],
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareRecomputeDropout(self, number_of_io_tiles):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def stage(x, name):
      with variable_scope.variable_scope(name, use_resource=True):
        weight = variable_scope.get_variable(
            "w",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      x = math_ops.matmul(x, weight)
      x = rand_ops.dropout(x, seed=[10, 10])
      return x

    def stage1(x, label):
      return stage(x, "s1"), label

    def stage2(x, label):
      return stage(x, "s2"), label

    def stage3(x, label):
      x = stage(x, "s3")
      logits = math_ops.reduce_sum(x, axis=[1])
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        recomp=True,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testUnmodifiedInput(self, number_of_io_tiles):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(8, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0]
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.1)

    def stage1(x, label):
      with variable_scope.variable_scope("s1", use_resource=True):
        weight = variable_scope.get_variable(
            "w",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      return weight + x, label

    def stage2(x, label):
      x = math_ops.reduce_sum(x, axis=[2])
      with variable_scope.variable_scope("s2", use_resource=True):
        weight = variable_scope.get_variable(
            "w",
            shape=[4, 8],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
      logits = math_ops.matmul(x, weight)
      return logits, label

    def stage3(logits, label):
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        10153,
        recomp=True,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineRecomputationCheckpoint(self, number_of_io_tiles):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def get_weight(name):
      return variable_scope.get_variable(
          name,
          shape=[4, 4],
          dtype=np.float32,
          initializer=init_ops.ones_initializer())

    def get_stage(x):
      w0 = get_weight("w0")
      w1 = get_weight("w1")
      x = math_ops.matmul(x, w0)
      x = nn.relu(x)
      x = pipelining_ops.recomputation_checkpoint(x)
      x = math_ops.matmul(x, w1)
      x = nn.relu(x)
      return x

    def stage1(x, label):
      with variable_scope.variable_scope("s1", use_resource=True):
        return get_stage(x), label

    def stage2(x, label):
      with variable_scope.variable_scope("s2", use_resource=True):
        return get_stage(x), label

    def stage3(x, label):
      logits = math_ops.reduce_mean(x, axis=[1])
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3],
        inputs_fn, [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        recomp=True,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        recomputation_mode=pipelining_ops.RecomputationMode.
        RecomputeAndBackpropagateInterleaved,
        number_of_io_tiles=number_of_io_tiles)

  @test_util.deprecated_graph_mode_only
  def testPipelineRecomputationCheckpointInFinalStage(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        return img, img

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def get_weight(name):
      return variable_scope.get_variable(
          name,
          shape=[2, 4, 4],
          dtype=np.float32,
          initializer=init_ops.ones_initializer())

    def stage1(x, label):
      return x, label

    def stage2(x, label):
      return x, label

    def stage3(x, label):
      # Create a chain of multiplies that we split in half with a checkpoint.
      # We expect only the first half to be recomputed in the bwd stage.
      with variable_scope.variable_scope("s3", use_resource=True):
        w0 = get_weight("w0")
        x = x * w0
        w1 = get_weight("w1")
        x = x * w1
        x = pipelining_ops.recomputation_checkpoint(x)
        w2 = get_weight("w2")
        x = x * w2
        w3 = get_weight("w3")
        x = x * w3
      return x + label

    def inputs_fn():
      with ops.device('cpu'):
        return []

    _, report_helper = \
        pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
            [stage1, stage2, stage3],
            inputs_fn, [],
            repeat_count,
            gradient_accumulation_count,
            dataset_fn,
            optimizer_fn,
            self,
            10535,
            recomp=True,
            schedule=pipelining_ops.PipelineSchedule.Grouped,
            recomputation_mode=pipelining_ops.RecomputationMode.
            RecomputeAndBackpropagateInterleaved,
            return_report=True)

    # With recomputation, we expect the following final forward stage
    #
    # inp0     inp1      inp2      ckpt      inp3      out
    # ---> mul ---> mul1 ---> ckpt ---> mul2 ---> mul3 --->
    #       ^        ^                   ^         ^
    #       |        |                   |         |
    #       w0       w1                  w2        w3
    #
    # to be split at the checkpoint, and only the first half to be recomputed in
    # the backward stage.
    # This is because:
    # 1. There is no point recomputing the second half, since we're about to use
    #    the computed activations in the backward pass immediately.
    # 2. Therefore anything past the last checkpoint should not be recomputed.
    # 3. We then don't need to store inp0 and inp1 in live memory while the
    #    rest of the forward pass and the second half of backprop is calculated.
    # Under normal recomputation (without the checkpoint), we'd expect to see
    # three extra multiplies in the backward pass to recompute inp3, inp2 and
    # inp1. With the checkpoint, we expect to not do the third multiply, mul2,
    # to recompute inp3, since it's after the last checkpoint.
    # The forward stage multiplies are called:
    #  - s3/mul/multiply.*/Op/Multiply
    #  - s3/mul_1/multiply.*/Op/Multiply
    #  - s3/mul_2/multiply.*/Op/Multiply
    #  - s3/mul_3/multiply.*/Op/Multiply
    # When we recompute, the scopes are copied with a different ID, so we can
    # make sure the multiply s3/mul_2/multiply.*/Op/Multiply isn't there twice.
    report = pva.openReport(report_helper.find_report())
    self.assert_compute_sets_matches(report, "s3/mul_2/multiply.*/Op/Multiply",
                                     1)

  @parameterized.parameters([0, 32])
  @test_util.deprecated_graph_mode_only
  def testPipelineRecomputationThreeInput(self, number_of_io_tiles):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32), value[1][1]

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

    def get_weight(name):
      return variable_scope.get_variable(
          name,
          shape=[4, 4],
          dtype=np.float32,
          initializer=init_ops.ones_initializer())

    def get_stage(x):
      w0 = get_weight("w0")
      w1 = get_weight("w1")
      x = math_ops.matmul(x, w0)
      x = nn.relu(x)
      x = math_ops.matmul(x, w1)
      x = nn.relu(x)
      return x

    def stage1(x, label, scale):
      with variable_scope.variable_scope("s1", use_resource=True):
        return get_stage(x) * scale[0], label

    def stage2(x, label):
      with variable_scope.variable_scope("s2", use_resource=True):
        return get_stage(x), label

    def stage3(x, label):
      logits = math_ops.reduce_mean(x, axis=[1])
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3],
        inputs_fn, [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        21458,
        recomp=True,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        number_of_io_tiles=number_of_io_tiles)


if __name__ == "__main__":
  googletest.main()
