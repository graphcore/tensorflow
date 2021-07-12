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
import numpy as np

from absl.testing import parameterized

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu.config import MergeRemoteBuffersBehaviour
from tensorflow.python.ipu.tests import pipelining_test_util


class PipeliningTest(test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
      {
          'replication_factor': 1,
          'minimum_remote_tensor_size': 128,
          'replicated_optimizer_state_sharding': False,
          'merge_remote_buffers': MergeRemoteBuffersBehaviour.MERGE,
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 128,
          'replicated_optimizer_state_sharding': False,
          'merge_remote_buffers': MergeRemoteBuffersBehaviour.NO_MERGING,
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 0,
          'replicated_optimizer_state_sharding': True,
          'merge_remote_buffers': MergeRemoteBuffersBehaviour.MERGE,
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 0,
          'replicated_optimizer_state_sharding': [True, False],
          'merge_remote_buffers': MergeRemoteBuffersBehaviour.MERGE,
      })
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare1(self, replication_factor,
                           minimum_remote_tensor_size,
                           replicated_optimizer_state_sharding,
                           merge_remote_buffers):
    # Check there is enough IPUs for this test.
    num_ipus_in_test = 4
    tu.skip_if_not_enough_ipus(self, replication_factor * num_ipus_in_test)

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

    pipeline_depth = 18
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.98)

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
          x = conv(x, 3, stride, out_filters, 'conv_1')
          x = nn.relu(x)

        with variable_scope.variable_scope(name + "/" + str(i) + "/2"):
          x = conv(x, 3, 1, out_filters, 'conv_2')

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

    def conv(x, ksize, stride, filters_out, name=None):
      return layers.Conv2D(
          filters_out,
          ksize,
          stride,
          'SAME',
          name=name,
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
            nn.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                        labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3],
        lambda: [], [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        optimizer_fn,
        self,
        1000,
        device_mapping=[0, 1, 2],
        replication_factor=replication_factor,
        schedule=pipelining_ops.PipelineSchedule.Interleaved,
        minimum_remote_tensor_size=minimum_remote_tensor_size,
        replicated_optimizer_state_sharding=replicated_optimizer_state_sharding,
        merge_remote_buffers=merge_remote_buffers)

  @parameterized.parameters(
      {
          'replication_factor': 1,
          'minimum_remote_tensor_size': 128,
          'offload_activations': False
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 128,
          'offload_activations': False
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 0,
          'offload_activations': False
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 0,
          'offload_activations': True
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 0,
          'offload_activations': True,
          'replicated_optimizer_state_sharding': False
      }, {
          'replication_factor': 2,
          'minimum_remote_tensor_size': 0,
          'offload_activations': True,
          'replicated_optimizer_state_sharding': [True, False],
      })
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights(
      self,
      replication_factor,
      minimum_remote_tensor_size,
      offload_activations,
      replicated_optimizer_state_sharding=None):
    # Check there is enough IPUs for this test.
    num_ipus_in_test = 4
    tu.skip_if_not_enough_ipus(self, replication_factor * num_ipus_in_test)

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    pipeline_depth = 20
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
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w1",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage4(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w3",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage5(x, label):
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

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3, stage4, stage5],
        inputs_fn, [10.01],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        optimizer_fn,
        self,
        1000,
        device_mapping=[0, 1, 2, 3, 0],
        replication_factor=replication_factor,
        schedule=pipelining_ops.PipelineSchedule.Grouped,
        minimum_remote_tensor_size=minimum_remote_tensor_size,
        offload_activations=offload_activations,
        replicated_optimizer_state_sharding=replicated_optimizer_state_sharding
    )


if __name__ == "__main__":
  googletest.main()
