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

import unittest
from absl.testing import parameterized
from functools import partial
import numpy as np
import pva

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training import optimizer as optimizer_lib
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import normalization_ops
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.config import IPUConfig
from tensorflow.python.ipu import gradient_accumulation as ga
from tensorflow.python.ipu.optimizers import map_gradient_optimizer
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()

DYNAMIC_ACCUMULATION_COUNT_CASES = [{
    'testcase_name': 'Fixed',
    'dynamic_in': False
}, {
    'testcase_name': 'Dynamic',
    'dynamic_in': True
}]


class PipeliningTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  @test_util.deprecated_graph_mode_only
  def testNoComputeGradsArgsWithV2(self):
    with self.assertRaisesRegex(
        ValueError,
        "OptimizerFunctionOutput.compute_gradients_args may not be used "
        "with OptimizerV2 instances."):
      opt = gradient_descent_v2.SGD()
      loss = math_ops.square(1)
      compute_args = (1, 2, 3)
      _ = pipelining_ops.OptimizerFunctionOutput(
          opt, loss, compute_gradients_args=compute_args)

  @test_util.deprecated_graph_mode_only
  def testNoComputeGradsKwargsWithV2(self):
    with self.assertRaisesRegex(
        ValueError,
        "OptimizerFunctionOutput.compute_gradients_kwargs may not be used "
        "with OptimizerV2 instances."):
      opt = gradient_descent_v2.SGD()
      loss = math_ops.square(1)
      compute_kwargs = {'a': 1, 'b': 2}
      _ = pipelining_ops.OptimizerFunctionOutput(
          opt, loss, compute_gradients_kwargs=compute_kwargs)

  @test_util.deprecated_graph_mode_only
  def testInvalidTypeForTape(self):
    with self.assertRaisesRegex(
        TypeError, "OptimizerFunctionOutput.tape must be a GradientTape."):
      opt = gradient_descent_v2.SGD()
      loss = math_ops.square(1)
      _ = pipelining_ops.OptimizerFunctionOutput(opt,
                                                 loss,
                                                 tape=['a', 'b', 'c'])

  @test_util.deprecated_graph_mode_only
  def testNoGradientTapeWithV1(self):
    with self.assertRaisesRegex(
        ValueError,
        "OptimizerFunctionOutput.tape may only be used with OptimizerV2."):
      opt = gradient_descent.GradientDescentOptimizer(1)
      with GradientTape() as tape:
        loss = math_ops.square(1)
        _ = pipelining_ops.OptimizerFunctionOutput(opt, loss, tape=tape)

  @test_util.deprecated_graph_mode_only
  def testNoVariablesWithV1(self):
    with self.assertRaisesRegex(
        ValueError,
        "OptimizerFunctionOutput.variables may only be used with OptimizerV2."
    ):
      opt = gradient_descent.GradientDescentOptimizer(1)
      loss = math_ops.square(1)
      _ = pipelining_ops.OptimizerFunctionOutput(opt,
                                                 loss,
                                                 variables=[1, 2, 3])

  @test_util.deprecated_graph_mode_only
  def testNoTapeWithVariables(self):
    with self.assertRaisesRegex(
        ValueError, "OptimizerFunctionOutput.tape may not be used when "
        "OptimizerFunctionOutput.variables is nonempty."):
      opt = gradient_descent_v2.SGD(1)
      with GradientTape() as tape:
        loss = math_ops.square(1)
        _ = pipelining_ops.OptimizerFunctionOutput(opt,
                                                   loss,
                                                   variables=[1, 2, 3],
                                                   tape=tape)

  @test_util.deprecated_graph_mode_only
  def testNoVariablesWithTape(self):
    with self.assertRaisesRegex(
        ValueError, "OptimizerFunctionOutput.variables must be empty when "
        "OptimizerFunctionOutput.tape is used."):
      opt = gradient_descent_v2.SGD(1)
      with GradientTape() as tape:
        loss = math_ops.square(1)
        f = pipelining_ops.OptimizerFunctionOutput(opt, loss, tape=tape)
        f.variables = [1, 2, 3]

  @test_util.deprecated_graph_mode_only
  def testPipelineNoOutfeedInference(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = x + 1
        return y

    def stage2(x):
      loss = math_ops.reduce_sum(x)
      return loss

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(
          ValueError, 'The last computational stage has tensor outputs'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineNoOutfeedWithOutputsTraining(self):
    def stage1(x):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = x + 1
        return y

    def stage2(x):
      y = layers.Conv2D(2,
                        1,
                        use_bias=True,
                        bias_initializer=init_ops.ones_initializer(),
                        kernel_initializer=init_ops.ones_initializer())(x)
      loss = math_ops.reduce_sum(y)
      return loss

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          optimizer_function=optimizer_function,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Grouped,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError,
                                  'The last computational stage has tensor'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineIterationsNotMultiple(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          10,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with tu.ipu_session() as sess:

      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 4
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          'The number of iterations of the pipeline must be a multiple of 3'):
        sess.run(r, {c: 10.01})

  @test_util.deprecated_graph_mode_only
  def testRTSButNoOffloading(self):
    """
    Sharding the offloaded optimizer state doesn't make sense when the
    optimizer state is not offloaded. This combination is invalid, so make sure
    it throws an error which tells the user it needs to be offloaded.
    """
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(**kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"]

    def stage2(x):
      return math_ops.reduce_sum(x)

    def stage3(x):
      return x

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def my_net():
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          3,
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=[2, 0, 1],
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          # Invalid combination. RTS only operates on offloaded state.
          replicated_optimizer_state_sharding=True,
          offload_weight_update_variables=False,
          optimizer_function=optimizer_function,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with self.assertRaisesRegex(ValueError,
                                'optimizer state must be offloaded'):
      with ops.device("/device:IPU:0"):
        ipu_compiler.compile(my_net)

  @test_util.deprecated_graph_mode_only
  def testPipelineInvalidDeviceMapping(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    # Wrong type:
    with self.assertRaisesRegex(
        TypeError, 'device_mapping argument needs to be a list or a tuple'):
      pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          3,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=1,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    # Too many values:
    with self.assertRaisesRegex(ValueError,
                                'Each stage must be mapped to an IPU'):
      pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          3,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=list(range(4)),
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    # Not enough values:
    with self.assertRaisesRegex(ValueError,
                                'Each stage must be mapped to an IPU'):
      pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          3,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=tuple(range(1)),
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMapping(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
    device_mapping = [2, 0, 1]

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          12,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=device_mapping,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with tu.ipu_session() as sess:

      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 4
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      tu.enable_ipu_events(cfg)
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ])
      report_json.parse_log()
      report_json.assert_pipeline_stages_on_expected_ipu(
          device_mapping, cfg.ipu_model.tiles_per_ipu)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMappingSameIpu(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
    device_mapping = [2, 2, 2]

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          12,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          device_mapping=device_mapping,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with tu.ipu_session() as sess:

      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 4
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      tu.enable_ipu_events(cfg)
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ])
      report_json.parse_log()
      report_json.assert_pipeline_stages_on_expected_ipu(
          device_mapping, cfg.ipu_model.tiles_per_ipu)

  @parameterized.named_parameters(*DYNAMIC_ACCUMULATION_COUNT_CASES)
  @test_util.deprecated_graph_mode_only
  def testPipelineWithInfeedsKwargs(self, dynamic_in):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(c, **kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"], c

    def stage2(x, c):
      return math_ops.reduce_sum(x) + c

    def stage3(x):
      return x

    def my_net(c, count):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          count,
          inputs=[c],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])
      count_ = array_ops.placeholder(np.int32, shape=[])
      count = count_ if dynamic_in else 12

    with tu.ipu_session() as sess:

      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c, count])

      cfg = IPUConfig()
      cfg.auto_select_ipus = 4
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      tu.enable_ipu_events(cfg)
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()

      report_json = tu.ReportJSON(self, sess)
      report_json.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      report_json.parse_log()
      sess.run(r, {c: 10.01, count_: 12})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ])
      report_json.parse_log()
      report_json.assert_pipeline_stages_on_expected_ipu(
          (0, 1, 3), cfg.ipu_model.tiles_per_ipu)

  @test_util.deprecated_graph_mode_only
  def testIllegalCapture(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    with ops.device('cpu'):
      y = array_ops.placeholder(np.float32, shape=[])

    def stage1(x):
      return x * y

    def stage2(x):
      return x

    def model_pipeline(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError, 'Trying to capture the tensor'):
        ipu_compiler.compile(model_pipeline, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testPipelineOnlyOneStage(self):
    def stage1(x):
      return x

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1],
          10,
          inputs=[x],
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError,
                                  'Pipeline requires at least two'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testDuplicateInputsOutputs(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(x, y):
      return x, y, y, x

    # The above should be optimised to a single copy for each duplicate output.
    def stage2(x1, y1, y2, x2):
      return x1, y1, y2, x2

    # Same for this stage
    def stage3(_x1, _y1, y2, x2):
      return x2, y2

    def model_pipeline(x, y):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          12,
          inputs=[x, y],
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[1, 2])

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(model_pipeline,
                                                     inputs=[x, y])

    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()
    utils.move_variable_initialization_to_cpu()

    #TODO(T10784) test how many IPU copies are here once we insert IPU copies.
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(compiled_model_pipeline, {
          x: np.ones(x.shape),
          y: np.ones(y.shape)
      })
      output = sess.run(outfeed_op)
      for i in range(12):
        self.assertAllClose(output[0][i], np.ones(x.shape))
        self.assertAllClose(output[1][i], np.ones(y.shape))

  @parameterized.named_parameters(*DYNAMIC_ACCUMULATION_COUNT_CASES)
  @test_util.deprecated_graph_mode_only
  def testPipelineWithStagesWithConstants(self, dynamic_in):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      idx = value[0][0][0][0]
      return {"a": a, "b": b, "idx": idx}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(c, **kwargs):
      y = layers.Conv2D(2,
                        1,
                        use_bias=True,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv1')(kwargs["a"])
      y = normalization_ops.group_norm(y)
      return y + kwargs["b"], c, kwargs["idx"]

    def stage2(x, c, idx):
      return x, c, idx

    def stage3(x, c, idx):
      return layers.Dense(
          2,
          kernel_initializer=init_ops.ones_initializer(),
          bias_initializer=init_ops.ones_initializer())(x), c, idx

    def stage4(x, c, idx):
      return math_ops.reduce_sum(
          layers.Dense(
              2,
              kernel_initializer=init_ops.ones_initializer(),
              bias_initializer=init_ops.ones_initializer())(x)) + c, idx

    def optimizer_function(loss, _):
      def func(grad, _):
        return clip_ops.clip_by_value(grad, -1., 1.)

      opt = map_gradient_optimizer.MapGradientOptimizer(
          gradient_descent.GradientDescentOptimizer(0.01), func)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def my_net(c, count):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3, stage4],
          count,
          inputs=[c],
          optimizer_function=optimizer_function,
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          reduction_method=ga.GradientAccumulationReductionMethod.SUM)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])
      count_ = array_ops.placeholder(np.int32, shape=[])
      count = count_ if dynamic_in else 12

    with tu.ipu_session() as sess:
      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net, inputs=[c, count])

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      tu.move_variable_initialization_to_cpu()
      outfeed_op = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      # Run the pipeline twice.
      sess.run(r, {c: 10.01, count_: 12})
      sess.run(r, {c: 10.01, count_: 12})
      losses_pipeline = sess.run(outfeed_op)
      # The values have been verified and compared against running the same
      # graph but sharded with gradient accumulation for 12 mini batches.
      self.assertAllClose(losses_pipeline[0], [
          1546.01, 1802.01, 1738.01, 1674.01, 1930.01, 1546.01, 1802.01,
          1738.01, 1674.01, 1930.01, 1546.01, 1802.01, 1331.1415, 1281.5806,
          1479.8259, 1182.457, 1380.7043, 1331.1415, 1281.5806, 1479.8259,
          1182.457, 1380.7043, 1331.1415, 1281.5806
      ])
      self.assertAllClose(losses_pipeline[1], [
          0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4, 1, 3, 0, 2, 4,
          1
      ])

  @test_util.deprecated_graph_mode_only
  def testPipelineWithStagesNoVariables(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[1])
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(features):
      partial = features * features
      return partial

    def stage2(partial):
      prediction = partial + partial
      return prediction

    def stage3(partial):
      return partial

    def model():
      with variable_scope.variable_scope("vs", use_resource=True):
        pipeline_op = pipelining_ops.pipeline(
            computational_stages=[stage1, stage2, stage3],
            gradient_accumulation_count=6,
            repeat_count=1,
            inputs=[],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            name="Pipeline",
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)
      return pipeline_op

    with tu.ipu_session() as sess:
      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(model, inputs=[])

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      tu.move_variable_initialization_to_cpu()
      outfeed_op = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      # Run the pipeline.
      sess.run(r)
      results = sess.run(outfeed_op)
      self.assertAllClose(results, [[0.], [2.], [8.], [18.], [32.], [0.]])

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          opt_cfg=[
              (gradient_descent.GradientDescentOptimizer, (0.01,)),
              (gradient_descent_v2.SGD, (0.01,)),
          ],
          dtype=[dtypes.float16, dtypes.float32],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare1(self, opt_cfg, reduction_method, dtype):
    opt_type, opt_args = opt_cfg

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7,
                                                    shape=[4, 4, 2],
                                                    dtype=dtype)
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        img = value / 7
        label = value[0][0][0][0]
        return img, label

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 20
    repeat_count = 2

    def optimizer_fn():
      return opt_type(*opt_args)

    # lr = 0.01
    # if reduction_method != ga.GradientAccumulationReductionMethod.SUM:
    #   lr /= 20
    # optimizer = gradient_descent.GradientDescentOptimizer(lr)

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
        return [array_ops.placeholder(dtype, shape=[])]

    rtol = 1e-6 if dtype == dtypes.float32 else 2e-3

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [10.01],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        15500,
        schedule=pipelining_ops.PipelineSchedule.Interleaved,
        reduction_method=reduction_method,
        rtol=rtol)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          opt_cfg=[
              (gradient_descent.GradientDescentOptimizer, (0.01,)),
              (gradient_descent_v2.SGD, (0.01,)),
          ],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare2(self, opt_cfg, reduction_method):
    # Resnet like network.
    opt_type, opt_args = opt_cfg

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
    repeat_count = 2

    def optimizer_fn():
      return opt_type(*opt_args)

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
        x = fc(x, 50)
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                            labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        38555,
        schedule=pipelining_ops.PipelineSchedule.Interleaved,
        reduction_method=reduction_method)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          opt_cfg=[
              (gradient_descent.GradientDescentOptimizer, (0.01,)),
              (gradient_descent_v2.SGD, (0.01,)),
          ],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @test_util.deprecated_graph_mode_only
  def testPipelineCompare3(self, opt_cfg, reduction_method):
    opt_type, opt_args = opt_cfg

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

    gradient_accumulation_count = 20
    repeat_count = 2

    def optimizer_fn():
      return opt_type(*opt_args)

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
        12600,
        schedule=pipelining_ops.PipelineSchedule.Interleaved,
        reduction_method=reduction_method)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          opt_cfg=[
              (gradient_descent.GradientDescentOptimizer, (0.01,)),
              (gradient_descent_v2.SGD, (0.01,)),
          ],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights(self, opt_cfg, reduction_method):
    opt_type, opt_args = opt_cfg

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(7, shape=[4, 4])

      def dataset_parser(value):
        img = value
        label = value[0][0] % 4
        return img, math_ops.cast(label, np.int32)

      dataset = dataset.map(dataset_parser)

      return dataset.batch(batch_size=2, drop_remainder=True)

    gradient_accumulation_count = 20
    repeat_count = 2

    def optimizer_fn():
      return opt_type(*opt_args)

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
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                            labels=label))
        return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    with self.assertRaisesRegex(NotImplementedError,
                                "The pipelining schedule"):
      pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
          [stage1, stage2, [stage3, stage4], stage5],
          inputs_fn, [10.01],
          repeat_count,
          gradient_accumulation_count,
          dataset_fn,
          optimizer_fn,
          self,
          21458,
          schedule=pipelining_ops.PipelineSchedule.Interleaved,
          device_mapping=[0, 1, [2, 3], 0],
          reduction_method=reduction_method)

  @test_util.deprecated_graph_mode_only
  def testStageOptionsNotEnough(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    with ops.device('cpu'):
      y = array_ops.placeholder(np.float32, shape=[])

    def stage1(x):
      return x * y

    def stage2(x):
      return x

    def model_pipeline(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          forward_propagation_stages_poplar_options=[
              pipelining_ops.PipelineStageOptions()
          ],
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(
          ValueError,
          'forward_propagation_stages_poplar_options must be a list or a tuple'
      ):
        ipu_compiler.compile(model_pipeline, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testStageOptionsWUWrongType(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      idx = value[0][0][0][0]
      return {"a": a, "b": b, "idx": idx}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(c, **kwargs):
      y = layers.Conv2D(2,
                        1,
                        use_bias=True,
                        kernel_initializer=init_ops.ones_initializer(),
                        name='conv1')(kwargs["a"])
      y = normalization_ops.group_norm(y)
      return y + kwargs["b"], c, kwargs["idx"]

    def stage2(x, c, idx):
      return x, c, idx

    def stage3(x, c, idx):
      return layers.Dense(
          2,
          kernel_initializer=init_ops.ones_initializer(),
          bias_initializer=init_ops.ones_initializer())(x), c, idx

    def stage4(x, c, idx):
      return math_ops.reduce_sum(
          layers.Dense(
              2,
              kernel_initializer=init_ops.ones_initializer(),
              bias_initializer=init_ops.ones_initializer())(x)) + c, idx

    def optimizer_function(loss, _):
      def func(grad, _):
        return clip_ops.clip_by_value(grad, -1., 1.)

      opt = map_gradient_optimizer.MapGradientOptimizer(
          gradient_descent.GradientDescentOptimizer(0.01), func)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def my_net(c):
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3, stage4],
          12,
          inputs=[c],
          optimizer_function=optimizer_function,
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved,
          weight_update_poplar_options={"dead": "beaf"},
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(
          TypeError,
          'weight_update_poplar_options to be of type PipelineStageOptions'):
        ipu_compiler.compile(my_net, inputs=[c])

  @test_util.deprecated_graph_mode_only
  def testOutfeedLossRequiresOutfeedAndOptimizerFunction(self):
    def identity(x):
      return x

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    with ops.device("/device:IPU:0"):
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
      with self.assertRaisesRegex(ValueError,
                                  "An optimizer_function must be provided"):
        pipelining_ops.pipeline(
            [identity, identity, identity, identity],
            gradient_accumulation_count=4,
            inputs=[1.0],
            outfeed_queue=outfeed_queue,
            outfeed_loss=True,
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with self.assertRaisesRegex(ValueError,
                                  "An outfeed_queue must be provided"):
        pipelining_ops.pipeline(
            [identity, identity, identity, identity],
            gradient_accumulation_count=4,
            inputs=[1.0],
            optimizer_function=optimizer_function,
            outfeed_loss=True,
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

  @test_util.deprecated_graph_mode_only
  def testOutfeedLoss(self):

    with tu.ipu_session() as sess:

      def stage1(_, x):
        with variable_scope.variable_scope("stage1", use_resource=True):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          return w * x

      def identity(x):
        return x

      def optimizer_function(x):
        opt = gradient_descent.GradientDescentOptimizer(0.01)
        loss = x + 1.0
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline(
            [stage1, identity, identity, identity],
            gradient_accumulation_count=8,
            inputs=[10, x],
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            outfeed_loss=True,
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[0.0])

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      self.assertAllEqual(np.ones(8), sess.run(outfed))

  @test_util.deprecated_graph_mode_only
  def testOutfeedMaskRequiresOutfeedAndOptimizerFunction(self):
    def identity(x):
      return x

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    with ops.device("/device:IPU:0"):
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
      with self.assertRaisesRegex(ValueError,
                                  "An optimizer_function must be provided"):
        pipelining_ops.pipeline(
            [identity, identity, identity, identity],
            gradient_accumulation_count=4,
            inputs=[1.0],
            outfeed_queue=outfeed_queue,
            outfeed_mask=[False],
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with self.assertRaisesRegex(ValueError,
                                  r".*no outfeed_queue has been provided"):
        pipelining_ops.pipeline(
            [identity, identity, identity, identity],
            gradient_accumulation_count=4,
            inputs=[1.0],
            optimizer_function=optimizer_function,
            outfeed_mask=[False],
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with self.assertRaisesRegex(
          ValueError, "Only one of `outfeed_loss` and "
          "`outfeed_mask` can be set."):
        pipelining_ops.pipeline(
            [identity, identity, identity, identity],
            gradient_accumulation_count=4,
            inputs=[1.0],
            optimizer_function=optimizer_function,
            outfeed_queue=outfeed_queue,
            outfeed_mask=[False],
            outfeed_loss=True,
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

  @test_util.deprecated_graph_mode_only
  def testOutfeedMask(self):

    with tu.ipu_session() as sess:

      def stage1(x):
        with variable_scope.variable_scope("stage1", use_resource=True):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          return x, w * x

      def stage(x, x2):
        return x, x2 + 1

      def optimizer_function(x, _):
        opt = gradient_descent.GradientDescentOptimizer(0.01)
        return pipelining_ops.OptimizerFunctionOutput(opt, x)

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline(
            [stage1, stage, stage, stage],
            gradient_accumulation_count=8,
            inputs=[x],
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            outfeed_mask=[True, False],
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0])

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.ipu_model.tiles_per_ipu = 2
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      self.assertAllEqual(np.full((1, 8), 4), sess.run(outfed))

  @test_util.deprecated_graph_mode_only
  def testConstantInput(self):

    with tu.ipu_session() as sess:

      def stage1(x):
        return 2 * x

      def stage2(x):
        return x + 1

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net():
        return pipelining_ops.pipeline(
            [stage1, stage2, stage2, stage2],
            gradient_accumulation_count=8,
            inputs=[42.0],
            outfeed_queue=outfeed_queue,
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net)

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = False
      cfg.ipu_model.tiles_per_ipu = 2
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()

      outfed = outfeed_queue.dequeue()
      sess.run(pipeline)

      expected = np.ones(8, dtype=np.float32) * 2 * 42 + 3
      actual = np.array(sess.run(outfed)).flatten()
      self.assertAllEqual(expected, actual)

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @test_util.deprecated_graph_mode_only
  def testOutfeedLossAccumulated(self, reduction_method):
    """ Tests accumulating the loss from the optimizer function. """
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      def stage1(x):
        with variable_scope.variable_scope("stage1", use_resource=True):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          return w * x

      def identity(x):
        return x

      def optimizer_function(x):
        opt = gradient_descent.GradientDescentOptimizer(0.01)
        loss = x + 1.0
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline([stage1, identity, identity, identity],
                                       gradient_accumulation_count=8,
                                       inputs=[x],
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=optimizer_function,
                                       outfeed_loss=True,
                                       accumulate_outfeed=True,
                                       reduction_method=reduction_method)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[0.0])

      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      # Loss of '1' is accumulated 8 times.
      self.assertAllEqual([8], sess.run(outfed))

    # There should be 2 GA-adds. One for the weight and one for the outfeed.
    report_json = pva.openReport(report_helper.find_report())
    ok = [
        'GradientAccumulatorAddWithScale', 'GradientAccumulatorAddWithScale_1'
    ]
    self.assert_compute_sets_contain_list(report_json, ok)

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @test_util.deprecated_graph_mode_only
  def testOutfeedAccumulatedTraining(self, reduction_method):
    """
    Tests accumulating an output from the last computational stage when
    training.
    """
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()
    with tu.ipu_session() as sess:

      def stage1(x):
        with variable_scope.variable_scope("stage1", use_resource=True):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          return w * x

      def identity(x):
        return x

      def optimizer_function(x):
        opt = gradient_descent.GradientDescentOptimizer(0.01)
        loss = x + 1.0
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline([stage1, identity, identity, identity],
                                       gradient_accumulation_count=8,
                                       inputs=[x],
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=optimizer_function,
                                       accumulate_outfeed=True,
                                       reduction_method=reduction_method)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0])

      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      # '1' is accumulated 8 times.
      self.assertAllEqual([[8]], sess.run(outfed))

    report_json = pva.openReport(report_helper.find_report())
    # There should be 2 GA-adds. One for the weight and one for the outfeed.
    ok = [
        'GradientAccumulatorAddWithScale', 'GradientAccumulatorAddWithScale_1'
    ]
    self.assert_compute_sets_contain_list(report_json, ok)

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @test_util.deprecated_graph_mode_only
  def testOutfeedAccumulatedTrainingSetDtype(self, reduction_method):
    """
    Tests accumulating a float16 loss, setting the accumulator dtype to float32
    to avoid overflow.
    """
    with tu.ipu_session() as sess:
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
      outfeed_queue2 = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(dtype, x):
        w_name = 'w1' if not dtype else 'w'
        outfeed = outfeed_queue if not dtype else outfeed_queue2

        def stage1(x):
          with variable_scope.variable_scope("stage1", use_resource=True):
            w = variable_scope.get_variable(name=w_name, initializer=1.0)
            return w * x

        def identity(x):
          return math_ops.cast(x + 10000, np.float16)

        def optimizer_function(x):
          opt = gradient_descent.GradientDescentOptimizer(0.01)
          loss = x + 1.0
          return pipelining_ops.OptimizerFunctionOutput(opt, loss)

        return pipelining_ops.pipeline([stage1, identity, identity, identity],
                                       gradient_accumulation_count=8,
                                       inputs=[x],
                                       outfeed_queue=outfeed,
                                       optimizer_function=optimizer_function,
                                       accumulate_outfeed=True,
                                       accumulate_outfeed_dtype=dtype,
                                       reduction_method=reduction_method)

      with ops.device("/device:IPU:0"):
        pipeline_16 = ipu_compiler.compile(partial(my_net, None), inputs=[1.0])
        pipeline_32 = ipu_compiler.compile(partial(my_net, np.float32),
                                           inputs=[1.0])

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()
      outfed2 = outfeed_queue2.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline_16)
      # Buffer overflows float16
      val = sess.run(outfed)[0]
      self.assertTrue(val > np.finfo(np.float16).max
                      or val < np.finfo(np.float16).min)

      sess.run(pipeline_32)
      # '1' is accumulated 8 times, + 24 ga count * 10000 addition to the loss
      val = sess.run(outfed2)[0]
      self.assertAllEqual([[240008]], [val])

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @test_util.deprecated_graph_mode_only
  def testOutfeedAccumulatedTrainingMultipleOutputs(self, reduction_method):
    """
    Tests accumulating two outputs from the last computational stage when
    training.
    """
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      def stage1(x, y):
        with variable_scope.variable_scope("stage1", use_resource=True):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          return w * x, y

      def identity(x, y):
        return x, y

      def optimizer_function(x, y):
        opt = gradient_descent.GradientDescentOptimizer(0.01)
        loss = x + y + 1.0
        return pipelining_ops.OptimizerFunctionOutput(opt, loss)

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x, y):
        return pipelining_ops.pipeline([stage1, identity, identity, identity],
                                       gradient_accumulation_count=8,
                                       inputs=[x, y],
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=optimizer_function,
                                       accumulate_outfeed=True,
                                       reduction_method=reduction_method)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0, 2.0])

      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      self.assertAllEqual([[8], [16]], sess.run(outfed))

    report_json = pva.openReport(report_helper.find_report())
    # There should be 3 GA-adds. One for the weight and one for each output.
    ok = [
        'GradientAccumulatorAddWithScale', 'GradientAccumulatorAddWithScale_1',
        'GradientAccumulatorAddWithScale_2'
    ]
    self.assert_compute_sets_contain_list(report_json, ok)

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @test_util.deprecated_graph_mode_only
  def testOutfeedAccumulatedInference(self, reduction_method):
    """ Tests accumulating an output from the last computational stage. """
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      def identity(x):
        return x

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline(
            [identity, identity, identity, identity],
            gradient_accumulation_count=8,
            inputs=[x],
            outfeed_queue=outfeed_queue,
            accumulate_outfeed=True,
            reduction_method=reduction_method)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0])

      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      # '1' is accumulated 8 times.
      self.assertAllEqual([[8]], sess.run(outfed))

    report_json = pva.openReport(report_helper.find_report())
    # There should be 1 GA-add for the outfeed.
    ok = ['GradientAccumulatorAddWithScale']
    self.assert_compute_sets_contain_list(report_json, ok)

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @test_util.deprecated_graph_mode_only
  def testOutfeedAccumulatedInferenceMultipleOutputs(self, reduction_method):
    """ Tests accumulating 2 outputs from the last computational stage. """
    cfg = IPUConfig()
    report_helper = tu.ReportHelper()
    report_helper.set_autoreport_options(cfg)
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()

    with tu.ipu_session() as sess:

      def identity(x, y):
        return x, y

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x, y):
        return pipelining_ops.pipeline(
            [identity, identity, identity, identity],
            gradient_accumulation_count=8,
            inputs=[x, y],
            outfeed_queue=outfeed_queue,
            accumulate_outfeed=True,
            reduction_method=reduction_method)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0, 2.0])

      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      # '1' is accumulated 8 times, '2' is accumulated 8 times.
      self.assertAllEqual([[8], [16]], sess.run(outfed))

    report_json = pva.openReport(report_helper.find_report())
    # There should be a GA-add for each output from the last stage.
    ok = [
        'GradientAccumulatorAddWithScale', 'GradientAccumulatorAddWithScale_1'
    ]
    self.assert_compute_sets_contain_list(report_json, ok)

  @test_util.deprecated_graph_mode_only
  def testOutfeedDictInference(self):

    with tu.ipu_session() as sess:

      def identity(x):
        return x

      def dictstage(x):
        return {"x": x}

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline(
            [identity, identity, identity, dictstage],
            gradient_accumulation_count=8,
            inputs=[x],
            outfeed_queue=outfeed_queue,
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0])

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      got = sess.run(outfed)
      self.assertIsInstance(got, dict)
      self.assertAllEqual(np.ones(8), got["x"])

  @test_util.deprecated_graph_mode_only
  def testOutfeedAccumulatedTrainingRequiresOutfeedALL(self):
    """
    Tests that the pipeline op requires a user to give an outfeed of mode ALL
    when accumulating the outfeed.
    """
    def stage1(x):
      with variable_scope.variable_scope("stage1", use_resource=True):
        w = variable_scope.get_variable(name="w", initializer=1.0)
        return w * x

    def identity(x):
      return x

    def optimizer_function(x):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      loss = x + 1.0
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue(
        outfeed_mode=ipu_outfeed_queue.IPUOutfeedMode.LAST)

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1, identity, identity, identity],
          gradient_accumulation_count=8,
          inputs=[x],
          outfeed_queue=outfeed_queue,
          optimizer_function=optimizer_function,
          accumulate_outfeed=True,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(
          ValueError,
          "To accumulate the outfeed, it must be in IPUOutfeedMode ALL."):
        ipu_compiler.compile(my_net, inputs=[1.0])

  @test_util.deprecated_graph_mode_only
  def testGradientShapeInference(self):

    with tu.ipu_session():

      variable_shape = (1, 2, 3)

      def stage1(x):
        with variable_scope.variable_scope("stage1", use_resource=True):
          w = variable_scope.get_variable(name="w", shape=variable_shape)
          return w * x

      def stage2(x):
        return x

      class MockOptimizer(gradient_descent.GradientDescentOptimizer):  # pylint: disable=abstract-method
        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
          self.applied_gradients = [g for (g, _) in grads_and_vars]
          return super().apply_gradients(grads_and_vars, global_step, name)

      optimizer = MockOptimizer(0.01)

      def optimizer_function(loss):
        return pipelining_ops.OptimizerFunctionOutput(optimizer, loss)

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline(
            [stage1, stage2],
            gradient_accumulation_count=4,
            inputs=[x],
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

      with ops.device("/device:IPU:0"):
        ipu_compiler.compile(my_net, inputs=[0.0])

      self.assertEqual(1, len(optimizer.applied_gradients))
      self.assertEqual(variable_shape, optimizer.applied_gradients[0].shape)

  @test_util.deprecated_graph_mode_only
  def testVariableInOptimizer(self):

    with tu.ipu_session() as sess:

      def stage1(x):
        with variable_scope.variable_scope("stage1", use_resource=True):
          w = variable_scope.get_variable(name="w", initializer=1.0)
          return w * x

      def identity(x):
        return x

      class MockOptimizer(gradient_descent.GradientDescentOptimizer):  # pylint: disable=abstract-method
        def __init__(self, lr):
          super(MockOptimizer, self).__init__(lr)
          with variable_scope.variable_scope("optimizer", use_resource=True):
            self.p = variable_scope.get_variable(name="p",
                                                 initializer=2.0,
                                                 trainable=False)

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
          grads_and_vars = [(g + self.p, v) for (g, v) in grads_and_vars]
          return super().apply_gradients(grads_and_vars, global_step, name)

      def optimizer_function(x):
        opt = MockOptimizer(0.5)
        return pipelining_ops.OptimizerFunctionOutput(opt, x)

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

      def my_net(x):
        return pipelining_ops.pipeline(
            [stage1, identity, identity, identity],
            gradient_accumulation_count=8,
            inputs=[x],
            outfeed_queue=outfeed_queue,
            optimizer_function=optimizer_function,
            outfeed_loss=True,
            reduction_method=ga.GradientAccumulationReductionMethod.SUM)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0])

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)

      # Accumulate 8 lots of gradient of 1.0 => 8.0, then add 2.0 then
      # apply LR and subtract from the original weight:
      #
      # 1.0 - (8.0 + 2.0) * 0.5 = -4.0
      for v in ops.get_default_graph().get_collection('variables'):
        if v.name == "stage1/w:0":
          new_v = sess.run(v)
          self.assertEqual(new_v, -4.0)

      # Now change the optimizer variable
      for v in ops.get_default_graph().get_collection('variables'):
        if v.name == "optimizer/p:0":
          sess.run(v.assign(4.0))

      sess.run(pipeline)

      # Accumulate 8 lots of gradient of 1.0 => -8.0, then add 30.0 then
      # apply LR and subtract from the original weight:
      #
      # -4.0 - (8.0 + 4.0) * 0.5 = -10.0
      for v in ops.get_default_graph().get_collection('variables'):
        if v.name == "stage1/w:0":
          new_v = sess.run(v)
          self.assertEqual(new_v, -10.0)

  @test_util.deprecated_graph_mode_only
  def testPipelineInferenceWithConditional(self):
    dataset = tu.create_single_increasing_dataset(10, shape=[1])
    dataset = dataset.batch(batch_size=1, drop_remainder=True)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(x):
      return x

    def stage2(x):
      return x

    def stage3(x):
      p = x > 2.0
      return control_flow_ops.cond(p, lambda: constant_op.constant(1.0),
                                   lambda: constant_op.constant(2.0))

    def my_net():
      return pipelining_ops.pipeline(
          [stage1, stage2, stage3],
          6,
          inputs=[],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with tu.ipu_session() as sess:
      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net)

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()
      sess.run(infeed_queue.initializer)
      sess.run(r)
      output = sess.run(outfeed_op)
      self.assertAllClose(output, [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])

  @test_util.deprecated_graph_mode_only
  def testPipelineWithCustomGradientFunction(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=4, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return value, math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    @custom_gradient.custom_gradient
    def f(x):
      x = x * x

      def grad(dy):
        return dy * x

      return x, grad

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      return x, label

    def stage2(x, label):
      return f(x), label

    def stage3(x, label):
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
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
        14415,
        schedule=pipelining_ops.PipelineSchedule.Grouped)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithLoop(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=4, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return value, math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      return x, label

    def stage2(x, label):
      x = control_flow_ops.while_loop(lambda i, _: i < 10,
                                      lambda i, x: (i + 1, x * x), (0, x),
                                      maximum_iterations=5)[1]

      return x, label

    def stage3(x, label):
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3], inputs_fn, [], repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer_fn, self, 11326)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithTensorArray(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=4, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int8), math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(x, label):
      x = math_ops.cast(x, np.float32)
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      return x, label

    def stage2(x, label):
      ta = tensor_array_ops.TensorArray(dtype=np.float32, size=4)

      def body(i, tx):
        tx = tx.write(i, math_ops.cast(i * 2, np.float32))
        return i + 1, tx

      ta = control_flow_ops.while_loop(lambda i, _: i < 4,
                                       body, (0, ta),
                                       maximum_iterations=5)[1]
      return x * ta.stack(), label

    def stage3(x, label):
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3], inputs_fn, [], repeat_count,
        gradient_accumulation_count, dataset_fn, optimizer_fn, self, 11326)

  @parameterized.parameters(list(ga.GradientAccumulationReductionMethod))
  @test_util.deprecated_graph_mode_only
  def testInvertPermutation(self, reduction_method):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=4, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int8), math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 24
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    def stage1(x, label):
      x = math_ops.cast(x, np.float32)
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable(
            "w2",
            shape=[4, 4],
            dtype=np.float32,
            initializer=init_ops.ones_initializer())
        x = math_ops.matmul(x, weight)
      x = array_ops.transpose(x, array_ops.invert_permutation([1, 0]))
      return x, label

    def stage2(x, label):
      return x, label

    def stage3(x, label):
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
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
        11326,
        reduction_method=reduction_method)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithEmbeddingOptimization(self):
    dataset_size = 100
    embedding_size = 15

    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(dataset_size, shape=[4])
      dataset = dataset.batch(batch_size=2, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return math_ops.cast(value,
                             np.int32), math_ops.cast(label % 4, np.int32)

      return dataset.map(dataset_parser)

    gradient_accumulation_count = 8
    repeat_count = 2

    def optimizer_fn():
      return gradient_descent.GradientDescentOptimizer(0.01)

    np.random.seed(1)
    embedding_shape = (dataset_size, embedding_size)
    embedding_initializer = np.random.normal(0, 1, embedding_shape).astype(
        np.float32)
    weights_shape = (embedding_size, embedding_size)
    weights_initializer = np.random.normal(0, 1,
                                           weights_shape).astype(np.float32)

    def stage1(idx, label):
      with variable_scope.variable_scope("stage1", use_resource=True):
        embedding = variable_scope.get_variable(
            "c",
            dtype=np.float32,
            initializer=embedding_initializer,
            trainable=True)
        x = embedding_ops.embedding_lookup(embedding, idx)
        return x, label

    def stage2(x, label):
      with variable_scope.variable_scope("vs", use_resource=True):
        weight = variable_scope.get_variable("w0",
                                             dtype=np.float32,
                                             initializer=weights_initializer,
                                             trainable=True)
        x = math_ops.matmul(x, weight)
      return x, label

    def stage3(x, label):
      x = math_ops.reduce_sum(x, axis=[-1])
      return x, label

    def stage4(x, label):
      loss = math_ops.reduce_mean(
          nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                          labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3, stage4],
        inputs_fn, [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        12049,
        schedule=pipelining_ops.PipelineSchedule.Interleaved)

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
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()
    grad_outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(features, labels):
      w = variable_scope.get_variable(name="w", initializer=initial_w)
      partial = w * features
      return partial, labels

    def stage2(partial, labels):
      loss = partial + labels
      return loss

    def identity(*args):
      return args

    def optimizer_function(loss):
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

      opt = CastingGradientDescent(self)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def model():
      pipeline_op = pipelining_ops.pipeline(
          computational_stages=[stage1, identity, identity, stage2],
          gradient_accumulation_count=gradient_accumulation_count,
          gradient_accumulation_dtype=gradient_accumulation_dtype,
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          optimizer_function=optimizer_function,
          name="Pipeline",
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)
      return pipeline_op

    def compiled_model():
      with ops.device("/device:IPU:0"):
        return ipu_compiler.compile(model)

    with tu.ipu_session() as sess:

      train_op = compiled_model()

      dequeued_gradient = grad_outfeed_queue.dequeue()

      cfg = IPUConfig()
      cfg.ipu_model.compile_ipu_code = True
      cfg.ipu_model.tiles_per_ipu = 128
      cfg.auto_select_ipus = 4
      cfg.configure_ipu_system()
      utils.move_variable_initialization_to_cpu()

      sess.run(infeed_queue.initializer)
      sess.run(variables.global_variables_initializer())

      sess.run(train_op)
      [actual_accumulated_gradient] = sess.run(dequeued_gradient)

      # L(x) = w * x + y
      # dL(x)/dw = x
      # This would overflow in fp16:
      expected_accumulated_gradient = x.astype(gradient_accumulation_dtype)

      self.assertAllEqual(expected_accumulated_gradient,
                          actual_accumulated_gradient)

      sess.run(infeed_queue.deleter)
      sess.run(outfeed_queue.deleter)
      sess.run(grad_outfeed_queue.deleter)

  @test_util.deprecated_graph_mode_only
  @tu.test_uses_ipus(num_ipus=4)
  def testGradientAccumulationDtypeTiedEmbedding(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    with ops.device('cpu'):
      indices = array_ops.placeholder(np.int32, [8])

    def stage1(indices):
      # Do an embedding lookup on a float16 embedding table.
      with variable_scope.variable_scope("vs", use_resource=True):
        table = variable_scope.get_variable(
            name="table",
            shape=[300, 300],
            dtype=dtypes.float16,
            initializer=init_ops.ones_initializer())
        return array_ops.gather(table, indices)

    def identity(*args):
      return args

    def stage2(partials):
      # Do a projection on the same float16 embeddding table.
      # Since the table has two (non-consecutive) pipeline stage users, and one
      # of those users is a valid AllocationFinder target, the gradient buffer
      # for the table will be allocated immediately in the DeferredVisitor.
      # When we accumulate in a different data type to the table, the buffer
      # should be allocated as the accumulating data type, not the table's data
      # type.
      with variable_scope.variable_scope("vs", use_resource=True, reuse=True):
        table = variable_scope.get_variable(
            name="table",
            shape=[300, 300],
            dtype=dtypes.float16,
            initializer=init_ops.ones_initializer())
        return math_ops.matmul(partials, table)

    def optimizer_function(loss):
      class CastingGradientDescent(optimizer_lib.Optimizer):  # pylint: disable=abstract-method
        """Compute update using the dtype of the gradient, and then cast to
        the dtype of the variable."""
        def __init__(self):
          super().__init__(use_locking=False, name="CastingGradientDescent")

        def apply_gradients(self, grads_and_vars, global_step=None, name=None):
          update_ops = []

          for (grad, var) in grads_and_vars:
            # Cast the gradient to be the var's dtype when applying in the WU.
            delta = math_ops.cast(-0.01 * grad, var.dtype)
            update_ops.append(var.assign_add(delta))

          return control_flow_ops.group(*update_ops)

      opt = CastingGradientDescent()
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def model():
      return pipelining_ops.pipeline(
          # There must be 4 stages here, otherwise:
          #  - there won't be >1 users of the gradient buffer because
          #  - both accs on the buffer will be on the same bwd stage since
          #  - the PipelineGradientAccumulationOptimizer didn't trigger because
          #  - it avoids putting size 0 FIFOs between consecutive stages.
          # a.k.a. the two stage users of the GA buffer can't be consecutive.
          computational_stages=[stage1, identity, identity, stage2],
          device_mapping=[0, 1, 1, 0],
          gradient_accumulation_count=8,
          # Accumulate the float16 embedding table's gradient in float32
          gradient_accumulation_dtype=dtypes.float32,
          inputs=[indices],
          outfeed_queue=outfeed_queue,
          optimizer_function=optimizer_function,
          name="Pipeline",
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device("/device:IPU:0"):
      train_op = ipu_compiler.compile(model)

    cfg = IPUConfig()
    cfg.ipu_model.compile_ipu_code = True
    cfg.ipu_model.tiles_per_ipu = 128
    cfg.auto_select_ipus = 4
    cfg.configure_ipu_system()
    utils.move_variable_initialization_to_cpu()

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(train_op, feed_dict={indices: np.ones([8], dtype=np.int32)})

  @test_util.deprecated_graph_mode_only
  def testPipeliningArgsAndKwargs(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(x):
      return x + 1

    def stage2(x):
      y = layers.Conv2D(2,
                        1,
                        use_bias=True,
                        bias_initializer=init_ops.ones_initializer(),
                        kernel_initializer=init_ops.ones_initializer())(x)
      loss = math_ops.reduce_sum(y)
      return loss

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      # Empty var list.
      compute_gradients_args = ([],)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss,
                                                    compute_gradients_args)

    def my_net(x):
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[x],
          outfeed_queue=outfeed_queue,
          optimizer_function=optimizer_function,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Grouped,
          reduction_method=ga.GradientAccumulationReductionMethod.MEAN)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError, 'No variables to optimize.'):
        ipu_compiler.compile(my_net, inputs=[x])

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          opt_cfg=[
              (gradient_descent.GradientDescentOptimizer, (0.01,)),
              (gradient_descent_v2.SGD, (0.01,)),
          ],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareMultiIPUStage(self, opt_cfg, reduction_method):
    # Resnet like network.
    opt_type, opt_args = opt_cfg

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
    repeat_count = 2

    def optimizer_fn():
      return opt_type(*opt_args)

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
        x = fc(x, 50)
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                            labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, stage2, stage3],
        lambda: [],
        [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        53362,
        device_mapping=[pipelining_ops._ALL_DEVICES, 0, 1],  # pylint: disable=W0212
        reduction_method=reduction_method)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          opt_cfg=[
              (gradient_descent.GradientDescentOptimizer, (0.01,)),
              (gradient_descent_v2.SGD, (0.01,)),
          ],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareParStages(self, opt_cfg, reduction_method):
    # Resnet like network.
    opt_type, opt_args = opt_cfg

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
    repeat_count = 2

    def optimizer_fn():
      return opt_type(*opt_args)

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

    def stage2a(x, _):
      with variable_scope.variable_scope("stage2a", use_resource=True):
        x = block("b", 2, 64, 1, x)
        return x

    def stage2b(x, label):
      with variable_scope.variable_scope("stage2b", use_resource=True):
        x = block("b", 2, 64, 1, x)
        return x, label

    def stage3(xa, xb, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        x = xa - xb
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = fc(x, 50)
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                            labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [stage1, [stage2a, stage2b], stage3],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        53362,
        device_mapping=[0, [0, 1], 1],
        reduction_method=reduction_method)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          opt_cfg=[
              (gradient_descent.GradientDescentOptimizer, (0.01,)),
              (gradient_descent_v2.SGD, (0.01,)),
          ],
          reduction_method=list(ga.GradientAccumulationReductionMethod)))
  @test_util.deprecated_graph_mode_only
  def testPipelineCompareParStagesInfeed(self, opt_cfg, reduction_method):
    # Resnet like network.
    opt_type, opt_args = opt_cfg

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
    repeat_count = 2

    def optimizer_fn():
      return opt_type(*opt_args)

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

    def stage1a(img, _):
      with variable_scope.variable_scope("stage1a", use_resource=True):
        x = conv(img, 7, 2, 16)
        x = nn.relu(x)
        x = max_pool(x, ksize=3, stride=2)
        return x

    def stage1b(img, label):
      with variable_scope.variable_scope("stage1b", use_resource=True):
        x = conv(img, 7, 2, 16)
        x = nn.softmax(x)
        x = max_pool(x, ksize=3, stride=2)
        return x, label

    def stage2(a, b, label):
      with variable_scope.variable_scope("stage2a", use_resource=True):
        x = block("b", 2, 64, 1, a + b)
        return x, label

    def stage3(x, label):
      with variable_scope.variable_scope("stage3", use_resource=True):
        x = math_ops.reduce_mean(x, axis=[1, 2])
        x = fc(x, 50)
        loss = math_ops.reduce_mean(
            nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                            labels=label))
        return loss

    pipelining_test_util.PipelineTester.compare_pipeline_to_sharding(
        [[stage1a, stage1b], stage2, stage3],
        lambda: [], [],
        repeat_count,
        gradient_accumulation_count,
        dataset_fn,
        optimizer_fn,
        self,
        61059,
        device_mapping=[[0, 1], 0, 1],
        reduction_method=reduction_method)

  def __makePipelineGATestNetwork(self, reduction_method):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset)
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue()

    def stage1(**kwargs):
      with variable_scope.variable_scope("vs", use_resource=True):
        y = layers.Conv2D(2,
                          1,
                          use_bias=True,
                          kernel_initializer=init_ops.ones_initializer(),
                          name='conv1')(kwargs["a"])
        return y + kwargs["b"]

    def stage2(x):
      return math_ops.reduce_sum(x)

    def optimizer_function(loss):
      opt = gradient_descent.GradientDescentOptimizer(0.01)
      return pipelining_ops.OptimizerFunctionOutput(opt, loss)

    def my_net():
      return pipelining_ops.pipeline(
          [stage1, stage2],
          10,
          inputs=[],
          infeed_queue=infeed_queue,
          outfeed_queue=outfeed_queue,
          optimizer_function=optimizer_function,
          pipeline_schedule=pipelining_ops.PipelineSchedule.Sequential,
          reduction_method=reduction_method)

    return my_net

  @test_util.deprecated_graph_mode_only
  def testPipelineGAReduceMethodNone(self):
    my_net = self.__makePipelineGATestNetwork(None)

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(
          ValueError,
          'Cannot parse None as one of GradientAccumulationReductionMethod.'):
        ipu_compiler.compile(my_net, inputs=[])

  @parameterized.parameters([
      'SUM', 'MEAN', 'RUNNING_MEAN',
      ga.GradientAccumulationReductionMethod.SUM,
      ga.GradientAccumulationReductionMethod.MEAN,
      ga.GradientAccumulationReductionMethod.RUNNING_MEAN
  ])
  @test_util.deprecated_graph_mode_only
  def testPipelineGAReduceMethodSupported(self, reduction_method):
    my_net = self.__makePipelineGATestNetwork(reduction_method)

    with ops.device("/device:IPU:0"):
      ipu_compiler.compile(my_net, inputs=[])

  @parameterized.parameters(['Exp', 10])
  @test_util.deprecated_graph_mode_only
  def testPipelineGAReduceMethodInvalid(self, reduction_method):
    my_net = self.__makePipelineGATestNetwork(reduction_method)

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(
          ValueError, f"Cannot parse {reduction_method} as one of "
          "GradientAccumulationReductionMethod."):
        ipu_compiler.compile(my_net, inputs=[])


if __name__ == "__main__":
  googletest.main()
