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
from functools import reduce
import operator

import numpy as np

from absl.testing import parameterized

from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.keras import layers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum
from tensorflow.python.training import optimizer
from tensorflow.python import ipu
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu.config import MergeRemoteBuffersBehaviour
from tensorflow.python.ipu.ops import cross_replica_ops
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
    optimizer = momentum.MomentumOptimizer(0.01, 0.98)

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
        optimizer,
        self,
        1000,
        device_mapping=[0, 1, 2],
        replication_factor=replication_factor,
        schedule=pipelining_ops.PipelineSchedule.Interleaved,
        minimum_remote_tensor_size=minimum_remote_tensor_size,
        replicated_optimizer_state_sharding=replicated_optimizer_state_sharding,
        merge_remote_buffers=merge_remote_buffers)

  @tu.test_uses_ipus(4, allow_ipu_model=False)
  @test_util.deprecated_graph_mode_only
  def testPipelineRTSLAMB(self):
    """
    Check that LAMB returns the same result when in and not in an RTS
    cluster, since the trust ratio should be replica-identical.
    """
    def dataset_fn():
      dataset = dataset_ops.Dataset.from_tensor_slices(
          np.arange(4 * 4 * 10).reshape(-1, 4, 4).astype(np.float32))
      dataset = dataset.repeat()

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)
      return dataset.batch(batch_size=2, drop_remainder=True)

    def _get_variable_name(param_name):
      """Get the variable name from the tensor name."""
      import re
      m = re.match("^(.*):\\d+$", param_name)
      if m is not None:
        param_name = m.group(1)
      return param_name

    class LAMBOptimizer(optimizer.Optimizer):
      #pylint: disable=W0223
      def __init__(self,
                   lr,
                   beta1=0.9,
                   beta2=0.999,
                   epsilon=1e-4,
                   name="LAMBOptimizer"):
        self.beta1 = math_ops.cast(beta1, dtypes.float32)
        self.beta2 = math_ops.cast(beta2, dtypes.float32)
        self.epsilon = math_ops.cast(epsilon, dtypes.float32)
        self.lr = lr
        super().__init__(False, name)

      def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        grads_and_vars = sorted(
            grads_and_vars, key=lambda x: reduce(operator.mul, x[0].shape, 1))
        for grad, weight in grads_and_vars:
          if grad is None or weight is None:
            continue

          # All-reduce the grads across the replicas so that the RTS clusters
          # receive replica-identical grads.
          grad = cross_replica_ops.cross_replica_sum(grad)

          m = variable_scope.get_variable(
              name=f"{_get_variable_name(weight.name)}/momentum",
              shape=weight.shape.as_list(),
              dtype=dtypes.float32,
              trainable=False,
              initializer=init_ops.zeros_initializer())
          v = variable_scope.get_variable(
              name=f"{_get_variable_name(weight.name)}/velocity",
              shape=weight.shape.as_list(),
              dtype=dtypes.float32,
              trainable=False,
              initializer=init_ops.zeros_initializer())
          grad = math_ops.cast(grad, dtype=dtypes.float32)
          # (Maybe loss scaling)
          next_m = self.beta1 * m + (1.0 - self.beta1) * grad
          next_v = self.beta2 * v + (1.0 - self.beta2) * grad**2
          update = next_m / (math_ops.sqrt(next_v) + self.epsilon)
          update = math_ops.cast(update, weight.dtype)
          # Norms
          w_norm = linalg_ops.norm(math_ops.cast(weight, dtype=dtypes.float32),
                                   ord=2)
          u_norm = linalg_ops.norm(update, ord=2)
          # Ratio
          ones = constant_op.constant(1.0,
                                      dtype=dtypes.float32,
                                      shape=w_norm.shape)
          ratio = array_ops.where(
              w_norm > 0, array_ops.where(u_norm > 0, w_norm / u_norm, ones),
              ones)
          # global -> layer-wise lr
          ratio = ratio * self.lr
          ratio = math_ops.cast(ratio, dtype=dtypes.float16)
          ratio = array_ops.reshape(ratio, shape=ratio.shape.as_list() + [1])
          # update term * layer-wise lr
          update = math_ops.cast(update, dtype=dtypes.float16)
          update = ratio * update
          update = math_ops.cast(update, dtype=weight.dtype)

          next_weight = weight - update
          assignments.extend(
              [weight.assign(next_weight),
               m.assign(next_m),
               v.assign(next_v)])
        return control_flow_ops.group(*assignments, name=name)

    def stage1(lr, x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w0",
            shape=[4, 8],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        return lr, x, label

    def stage2(lr, x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w1",
            shape=[8, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
        logits = math_ops.reduce_mean(x, axis=[1])
        loss = math_ops.reduce_mean(
            nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label))
        return lr, loss

    cfg = ipu.config.IPUConfig()
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.optimizations.minimum_remote_tensor_size = 0
    if tu.has_ci_ipus():
      tu.add_hw_ci_connection_options(cfg)
    cfg.optimizations.merge_remote_buffers = \
        ipu.config.MergeRemoteBuffersBehaviour.MERGE
    cfg.configure_ipu_system()
    ipu.utils.move_variable_initialization_to_cpu()

    def run(rts=True):
      g = ops.Graph()
      with g.as_default(), self.test_session(graph=g) as session:
        dataset = dataset_fn()
        infeed_queue = ipu.ipu_infeed_queue.IPUInfeedQueue(dataset)
        outfeed_queue = ipu.ipu_outfeed_queue.IPUOutfeedQueue()

        def opt_fn(lr, loss):
          opt = LAMBOptimizer(lr)
          return pipelining_ops.OptimizerFunctionOutput(opt, loss)

        def my_net(*args):
          with variable_scope.variable_scope("ipu",
                                             use_resource=True,
                                             reuse=False):
            return pipelining_ops.pipeline(
                [stage1, stage2],
                gradient_accumulation_count=8,
                repeat_count=10,
                inputs=args,
                optimizer_function=opt_fn,
                infeed_queue=infeed_queue,
                outfeed_queue=outfeed_queue,
                pipeline_schedule=pipelining_ops.PipelineSchedule.Grouped,
                recomputation_mode=None,
                device_mapping=[0, 1],
                replicated_optimizer_state_sharding=rts,
                offload_weight_update_variables=True)

        with ops.device("cpu"):
          lr = array_ops.placeholder(dtypes.float32, shape=[])

        with ipu.scopes.ipu_scope("/device:IPU:0"):
          run = ipu.ipu_compiler.compile(my_net, inputs=[lr])

        session.run(variables.global_variables_initializer())
        session.run(infeed_queue.initializer)
        session.run(run, feed_dict={lr: 0.01})
        return session.run(outfeed_queue.dequeue())

    self.assertAllEqual(run(rts=True), run(rts=False))

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
    optimizer = momentum.MomentumOptimizer(0.01, 0.98)

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
        inputs_fn, [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        optimizer,
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
