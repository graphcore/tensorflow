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

from tensorflow.python.ipu import test_utils as tu
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import gradient_accumulation as ga
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.python.ipu.utils import MergeRemoteBuffersBehaviour


class BatchSerialPipeliningHwTest(test.TestCase, parameterized.TestCase):
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testInference(self):
    pipeline_depth = 18
    repeat_count = 4
    size = pipeline_depth * repeat_count

    def dataset_fn():
      return tu.create_single_increasing_dataset(size, shape=[2])

    def stage(x):
      return x

    result = pipelining_test_util.PipelineTester.pipeline_on_ipu(
        [stage] * 5,
        lambda: [], [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        None,
        self,
        1000,
        False,
        device_mapping=[0] * 5,
        schedule=pipelining_ops.PipelineSchedule.Sequential,
        replication_factor=1,
        offload_activations=None)
    self.assertAllClose(np.reshape(result, [size, 2]),
                        np.mgrid[0:size, 0:2][0])

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          merge_remote_buffers=[
              MergeRemoteBuffersBehaviour.NO_MERGING,
              MergeRemoteBuffersBehaviour.MERGE
          ],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @tu.test_uses_ipus(num_ipus=1)
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare1(self, merge_remote_buffers, reduction_method):
    # Resnet like network.
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[32, 32, 4])

      def dataset_parser(value):
        return value, math_ops.cast(value[0][0][0], np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    pipeline_depth = 9
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(1.0 * 10**-6, 0.8)

    def fc(x, num_units_out):
      return layers.Dense(
          num_units_out,
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def max_pool(x, ksize=2, stride=None):
      return layers.MaxPooling2D(ksize, stride, padding='SAME')(x)

    def conv(x, filters_out, ksize, name=None):
      return layers.Conv2D(
          filters_out,
          ksize,
          name=name,
          padding='SAME',
          kernel_initializer=init_ops.constant_initializer(0.1),
          bias_initializer=init_ops.constant_initializer(0.0))(x)

    def stage1(img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        x = conv(img, 8, 3)
        x = nn.relu(x)
        x = max_pool(x)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = conv(x, 16, 3, name="conv1")
        x = nn.relu(x)
        x = max_pool(x)

        x = conv(x, 32, 3, name="conv2")
        x = nn.relu(x)
        x = max_pool(x)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = nn.relu(x)
        x = fc(x, 7)
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
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
        replication_factor=1,
        schedule=pipelining_ops.PipelineSchedule.Sequential,
        offload_activations=None,
        merge_remote_buffers=merge_remote_buffers,
        reduction_method=reduction_method)

  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[8, 8])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 8
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    pipeline_depth = 20
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(1.0 * 10**-6, 0.8)

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[8, 8],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w1",
            shape=[8, 8],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[8, 8],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage4(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w3",
            shape=[8, 8],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return x, label

    def stage5(x, label):
      # Ruse the weight here.
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[8, 8],
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

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3, stage4, stage5],
        inputs_fn, [10.01],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        optimizer_fn,
        self,
        1000,
        replication_factor=2,
        schedule=pipelining_ops.PipelineSchedule.Sequential,
        offload_activations=None,
        device_mapping=[0, 1, 1, 0, 0])


if __name__ == "__main__":
  googletest.main()
