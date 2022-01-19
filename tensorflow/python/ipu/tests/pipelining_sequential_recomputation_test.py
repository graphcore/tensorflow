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

import numpy as np
from absl.testing import parameterized
import pva

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import internal_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu.optimizers import gradient_accumulation_optimizer as ga
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


class PipeliningSeqRecomputationTest(test_util.TensorFlowTestCase,
                                     parameterized.TestCase):
  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare1(self):
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
        [stage1, stage2, stage3, stage4], inputs_fn, [10.01], repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer_fn, self, 13936,
        True, pipelining_ops.PipelineSchedule.Sequential)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare2(self):
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
        x = conv(img, 3, 2, 4)
        x = nn.relu(x)
        x = max_pool(x, ksize=3, stride=2)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = block("b", 2, 4, 1, x)
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
        [stage1, stage2, stage3], lambda: [], [], repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer_fn, self, 15190,
        True, pipelining_ops.PipelineSchedule.Sequential)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare3(self):
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
            nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4], lambda: [], [], repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer_fn, self, 13681,
        True, pipelining_ops.PipelineSchedule.Sequential)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare4(self):
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
        [stage1, stage2, stage3, stage4], inputs_fn, [10.01], repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer_fn, self, 10760,
        True, pipelining_ops.PipelineSchedule.Sequential)

  @tu.skip_on_hw
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare5(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(128, shape=[1, 1, 1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 8
    repeat_count = 2

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
        [stage1, stage2, stage3, stage4], inputs_fn, [10.01], repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer_fn, self, 6328,
        True, pipelining_ops.PipelineSchedule.Sequential)

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @tu.test_uses_ipus(num_ipus=2)
  @test_util.deprecated_graph_mode_only
  def testDistributedBatchNorm(self, reduction_method):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(128, shape=[1, 1, 1])
      dataset = dataset.batch(batch_size=1, drop_remainder=True)

      def dataset_parser(value):
        img = value
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 8
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(img, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        y = layers.Conv2D(
            2,
            1,
            use_bias=True,
            kernel_initializer=init_ops.constant_initializer(0.5),
            bias_initializer=init_ops.constant_initializer(0.5),
            name='conv1')(img)
        return y, label

    def stage2(x, label):
      with variable_scope.variable_scope("stage2", use_resource=True):
        x = layers.BatchNormalization()(x, training=True)
        x = x * x
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage4", use_resource=True):
        return math_ops.reduce_sum(x) + label

    def opt_fn(loss):
      return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset_fn())
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def my_net():
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          gradient_accumulation_count,
          repeat_count=repeat_count,
          optimizer_function=opt_fn,
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential,
          device_mapping=[0] * 3,
          reduction_method=reduction_method)

    with self.test_session() as session:
      cfg = IPUConfig()
      report_helper = tu.ReportHelper()
      report_helper.set_autoreport_options(cfg)
      cfg.optimizations.enable_graph_outlining = True
      cfg.auto_select_ipus = 2
      cfg.allow_recompute = True
      cfg.norms.experimental.distributed_batch_norm_replica_group_size = 2
      tu.add_hw_ci_connection_options(cfg)
      cfg.configure_ipu_system()

      with ops.device("/device:IPU:0"):
        compiled_model_pipeline = ipu_compiler.compile(my_net, inputs=[])

      utils.move_variable_initialization_to_cpu()

      session.run(variables.global_variables_initializer())
      session.run(infeed_queue.initializer)

      session.run(compiled_model_pipeline)

    # Check that there is no compute set for recomputed stats.
    # pylint: disable=line-too-long
    bad = [
        'stage2/batch_normalization/FusedBatchNorm*/Recomputed/batch-norm-statistics*'
    ]
    # pylint: enable=line-too-long
    report = pva.openReport(report_helper.find_report())
    self.assert_compute_sets_not_in_blacklist(report, bad)


if __name__ == "__main__":
  googletest.main()
