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
from abc import ABCMeta, abstractmethod

import numpy as np

import popdist

from tensorflow.python.ipu import test_utils as tu
from tensorflow.keras import layers
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compiler.plugin.poplar.ops import gen_poputil_ops
from tensorflow.python.framework import ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.ops import cross_replica_ops
from tensorflow.python.ipu.optimizers import IpuOptimizer


class BaseTestCrossReplicaOptimizer(IpuOptimizer, metaclass=ABCMeta):
  """ Base test optimizer that averages gradients across IPU replicas."""
  def __init__(self, opt, name="BaseTestCrossReplicaOptimizer"):
    super(BaseTestCrossReplicaOptimizer, self).__init__(opt, name=name)

  @abstractmethod
  def cross_replica_reduce(self, grad):
    pass

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    summed_grads_and_vars = []
    for (grad, var) in grads_and_vars:
      if grad is None:
        summed_grads_and_vars.append((grad, var))
      else:
        with ops.colocate_with(grad):
          summed_grads_and_vars.append((self.cross_replica_reduce(grad), var))

    return self._opt.apply_gradients(summed_grads_and_vars, global_step, name)


class CrossReplicaOptimizerWithCrossReplicaSumAndNorm(
    BaseTestCrossReplicaOptimizer):
  def cross_replica_reduce(self, grad):
    return gen_poputil_ops.ipu_replication_normalise(
        cross_replica_ops.cross_replica_sum(grad))


class CrossReplicaOptimizerWithCrossReplicaMean(BaseTestCrossReplicaOptimizer):
  def cross_replica_reduce(self, grad):
    return cross_replica_ops.cross_replica_mean(grad)


class PoprunReplicaPartitioningTest(test.TestCase):
  def _compare_partitioned_to_non_partitioned(self, stages, repeat_count,
                                              gradient_accumulation_count,
                                              dataset_fn, optimizer_fn):

    replication_factor = popdist.getNumLocalReplicas()
    ipu_id = popdist.getDeviceId()
    process_count = popdist.getNumInstances()
    process_index = popdist.getInstanceIndex()

    losses = dict()
    variables = dict()

    for partitioning in [False, True]:
      losses[partitioning], variables[
          partitioning] = pipelining_test_util.PipelineTester.pipeline_on_ipu(
              stages,
              lambda: [], [],
              repeat_count,
              gradient_accumulation_count,
              dataset_fn,
              optimizer_fn,
              test_wrapper=self,
              recomp=False,
              schedule=None,
              expected_max_tile_memory=1000,
              replication_factor=replication_factor,
              replicated_optimizer_state_sharding=partitioning,
              minimum_remote_tensor_size=0,
              return_vars=True,
              ipu_id=ipu_id,
              process_count=process_count,
              process_index=process_index)

    self.assertAllClose(losses[False], losses[True])
    self.assertAllClose(variables[False], variables[True])

  def _setupTest(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      # Make the data different in each instance:
      dataset = dataset.map(lambda x: x * (popdist.getInstanceIndex() + 1.0))
      dataset = dataset.batch(batch_size=16, drop_remainder=True)
      dataset = dataset.batch(batch_size=16, drop_remainder=True)
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = math_ops.reduce_mean(img, axis=[1, 2, 3])
        return img, math_ops.cast(label, np.int32)

      return dataset.map(dataset_parser)

    pipeline_depth = 8
    repeat_count = 2

    def optimizer_fn():
      return momentum.MomentumOptimizer(0.01, 0.5)

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
        x = conv(img, 5, 2, 8)
        x = nn.relu(x)
        x = max_pool(x, ksize=3, stride=2)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = fc(x, 10)
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                            labels=label))
        return loss

    return stage1, stage2, repeat_count, \
      pipeline_depth, dataset_fn, optimizer_fn

  @tu.test_uses_ipus(num_ipus=16)
  @test_util.deprecated_graph_mode_only
  def test_compare_partitioned_to_non_partitioned(self):
    stage1, stage2, repeat_count, \
      pipeline_depth, dataset_fn, optimizer_fn = self._setupTest()

    self._compare_partitioned_to_non_partitioned([stage1, stage2],
                                                 repeat_count, pipeline_depth,
                                                 dataset_fn, optimizer_fn)

  @tu.test_uses_ipus(num_ipus=16)
  @test_util.deprecated_graph_mode_only
  def test_partitioned_cross_replica_mean(self):
    stage1, stage2, repeat_count, \
      pipeline_depth, dataset_fn, optimizer_fn = self._setupTest()

    replication_factor = popdist.getNumLocalReplicas()
    ipu_id = popdist.getDeviceId()
    process_count = popdist.getNumInstances()
    process_index = popdist.getInstanceIndex()

    losses_ref, variables_ref = \
      pipelining_test_util.PipelineTester.pipeline_on_ipu(
          [stage1, stage2],
          lambda: [], [],
          repeat_count,
          pipeline_depth,
          dataset_fn,
          optimizer_fn,
          test_wrapper=self,
          recomp=False,
          schedule=None,
          expected_max_tile_memory=1000,
          replication_factor=replication_factor,
          replicated_optimizer_state_sharding=True,
          minimum_remote_tensor_size=0,
          return_vars=True,
          ipu_id=ipu_id,
          process_count=process_count,
          process_index=process_index,
          cross_replica_optimizer_cls=
          CrossReplicaOptimizerWithCrossReplicaSumAndNorm,
          scheduler=utils.SchedulingAlgorithm.SHORTEST_PATH)

    losses_test, variables_test = \
      pipelining_test_util.PipelineTester.pipeline_on_ipu(
          [stage1, stage2],
          lambda: [], [],
          repeat_count,
          pipeline_depth,
          dataset_fn,
          optimizer_fn,
          test_wrapper=self,
          recomp=False,
          schedule=None,
          expected_max_tile_memory=1000,
          replication_factor=replication_factor,
          replicated_optimizer_state_sharding=True,
          minimum_remote_tensor_size=0,
          return_vars=True,
          ipu_id=ipu_id,
          process_count=process_count,
          process_index=process_index,
          cross_replica_optimizer_cls=CrossReplicaOptimizerWithCrossReplicaMean,
          scheduler=utils.SchedulingAlgorithm.SHORTEST_PATH)

    self.assertAllClose(losses_ref, losses_test)
    self.assertAllClose(variables_ref, variables_test)


if __name__ == "__main__":
  googletest.main()
