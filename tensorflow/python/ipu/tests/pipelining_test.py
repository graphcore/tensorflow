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

from tensorflow.keras import layers
from tensorflow.compiler.plugin.poplar.tests import test_utils as tu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.ipu import embedding_ops
from tensorflow.python.ipu import ipu_compiler
from tensorflow.python.ipu import ipu_infeed_queue
from tensorflow.python.ipu import ipu_outfeed_queue
from tensorflow.python.ipu import normalization_ops
from tensorflow.python.ipu import pipelining_ops
from tensorflow.python.ipu import utils
from tensorflow.python.ipu.optimizers import map_gradient_optimizer
from tensorflow.python.ipu.tests import pipelining_test_util
from tensorflow.compat.v1 import disable_v2_behavior

disable_v2_behavior()


class PipeliningTest(test_util.TensorFlowTestCase):
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
      return pipelining_ops.pipeline([stage1, stage2], 10, inputs=[x])

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Grouped)

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
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed1")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed1")

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
    cfg = utils.set_ipu_model_options(cfg,
                                      compile_ipu_code=True,
                                      tiles_per_ipu=128)
    cfg = utils.auto_select_ipus(cfg, 4)
    utils.configure_ipu_system(cfg)
    utils.move_variable_initialization_to_cpu()

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      with self.assertRaisesRegex(
          errors.FailedPreconditionError,
          'The pipeline depth of the pipeline must be a multiple of 3'):
        sess.run(r, {c: 10.01})

  @test_util.deprecated_graph_mode_only
  def testPipelineInvalidDeviceMapping(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed3")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed3")

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMapping(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed4")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed4")
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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
    cfg = utils.set_ipu_model_options(cfg,
                                      compile_ipu_code=True,
                                      tiles_per_ipu=128)
    cfg = utils.auto_select_ipus(cfg, 4)
    utils.configure_ipu_system(cfg)
    utils.move_variable_initialization_to_cpu()

    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ]])
      report.parse_log()
      report.assert_pipeline_stages_on_expected_ipu(device_mapping)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithDeviceMappingSameIpu(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed5")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed5")
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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
    cfg = utils.set_ipu_model_options(cfg,
                                      compile_ipu_code=True,
                                      tiles_per_ipu=128)
    cfg = utils.auto_select_ipus(cfg, 4)
    utils.configure_ipu_system(cfg)
    utils.move_variable_initialization_to_cpu()

    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ]])
      report.parse_log()
      report.assert_pipeline_stages_on_expected_ipu(device_mapping)

  @test_util.deprecated_graph_mode_only
  def testPipelineWithInfeedsKwargs(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      return {"a": a, "b": b}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed6")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed6")

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
    cfg = utils.set_ipu_model_options(cfg,
                                      compile_ipu_code=True,
                                      tiles_per_ipu=128)
    cfg = utils.auto_select_ipus(cfg, 4)
    utils.configure_ipu_system(cfg)
    utils.move_variable_initialization_to_cpu()

    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      report = tu.ReportJSON(self, sess, configure_device=False)
      report.reset()
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      report.parse_log()
      sess.run(r, {c: 10.01})
      losses_pipeline = sess.run(outfeed_op)
      self.assertAllClose(losses_pipeline, [[
          410.01, 730.01, 650.01, 570.01, 890.01, 410.01, 730.01, 650.01,
          570.01, 890.01, 410.01, 730.01
      ]])
      report.parse_log()
      report.assert_pipeline_stages_on_expected_ipu((0, 1, 3))

  @test_util.deprecated_graph_mode_only
  def testIllegalCapture(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed8")

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])

    with ops.device("/device:IPU:0"):
      with self.assertRaisesRegex(ValueError,
                                  'Pipeline requires at least two'):
        ipu_compiler.compile(my_net, inputs=[x])

  @test_util.deprecated_graph_mode_only
  def testDuplicateInputsOutputs(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed9")

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

    with ops.device('cpu'):
      x = array_ops.placeholder(np.float32, shape=[1, 4, 4, 2])
      y = array_ops.placeholder(np.float32, shape=[1, 2])

    with ops.device("/device:IPU:0"):
      compiled_model_pipeline = ipu_compiler.compile(model_pipeline,
                                                     inputs=[x, y])

    cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
    cfg = utils.set_ipu_model_options(cfg,
                                      compile_ipu_code=True,
                                      tiles_per_ipu=128)
    cfg = utils.auto_select_ipus(cfg, 4)
    utils.configure_ipu_system(cfg)
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

  @test_util.deprecated_graph_mode_only
  def testPipelineWithStagesWithConstants(self):
    dataset = tu.create_single_increasing_dataset(5, shape=[4, 4, 2])
    dataset = dataset.batch(batch_size=2, drop_remainder=True)

    def dataset_parser(value):
      a = value
      b = (value + 10.) / 2.0
      idx = value[0][0][0][0]
      return {"a": a, "b": b, "idx": idx}

    dataset = dataset.map(dataset_parser)
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed10")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed10")

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
          pipeline_schedule=pipelining_ops.PipelineSchedule.Interleaved)

    with ops.device('cpu'):
      c = array_ops.placeholder(np.float32, shape=[])

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[c])

    cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
    cfg = utils.set_ipu_model_options(cfg,
                                      compile_ipu_code=True,
                                      tiles_per_ipu=128)
    cfg = utils.auto_select_ipus(cfg, 4)
    utils.configure_ipu_system(cfg)
    utils.move_variable_initialization_to_cpu()

    tu.move_variable_initialization_to_cpu()
    outfeed_op = outfeed_queue.dequeue()
    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      # Run the pipeline twice.
      sess.run(r, {c: 10.01})
      sess.run(r, {c: 10.01})
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
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed11")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed11")

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
            pipeline_depth=6,
            repeat_count=1,
            inputs=[],
            infeed_queue=infeed_queue,
            outfeed_queue=outfeed_queue,
            name="Pipeline")
      return pipeline_op

    with tu.ipu_session() as sess:
      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(model, inputs=[])

      cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
      cfg = utils.set_ipu_model_options(cfg,
                                        compile_ipu_code=True,
                                        tiles_per_ipu=128)
      cfg = utils.auto_select_ipus(cfg, 4)
      utils.configure_ipu_system(cfg)
      utils.move_variable_initialization_to_cpu()

      tu.move_variable_initialization_to_cpu()
      outfeed_op = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(infeed_queue.initializer)
      # Run the pipeline.
      sess.run(r)
      results = sess.run(outfeed_op)
      self.assertAllClose(results[0], [[0.], [2.], [8.], [18.], [32.], [0.]])

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

    pipeline_depth = 20
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

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
        pipeline_depth,
        dataset_fn,
        optimizer,
        self,
        15500,
        schedule=pipelining_ops.PipelineSchedule.Interleaved)

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

    pipeline_depth = 18
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
        38555,
        schedule=pipelining_ops.PipelineSchedule.Interleaved)

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

    pipeline_depth = 20
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

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
        [stage1, stage2, stage3, stage4],
        lambda: [], [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        optimizer,
        self,
        12600,
        schedule=pipelining_ops.PipelineSchedule.Interleaved)

  @test_util.deprecated_graph_mode_only
  def testPipelineCompareSharedWeights(self):
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

    with self.assertRaisesRegex(NotImplementedError,
                                "The pipelining schedule"):
      pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
          [stage1, stage2, stage3, stage4, stage5],
          inputs_fn, [10.01],
          repeat_count,
          pipeline_depth,
          dataset_fn,
          optimizer,
          self,
          21458,
          schedule=pipelining_ops.PipelineSchedule.Interleaved,
          device_mapping=[0, 1, 2, 3, 0])

  @test_util.deprecated_graph_mode_only
  def testStageOptionsNotEnough(self):
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed8")

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
          ])

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
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed10")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed10")

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
          weight_update_poplar_options={"dead": "beaf"})

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
      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed11")
      with self.assertRaisesRegex(ValueError,
                                  "An optimizer_function must be provided"):
        pipelining_ops.pipeline([identity, identity, identity, identity],
                                pipeline_depth=4,
                                inputs=[1.0],
                                outfeed_queue=outfeed_queue,
                                outfeed_loss=True)

      with self.assertRaisesRegex(ValueError,
                                  "An outfeed_queue must be provided"):
        pipelining_ops.pipeline([identity, identity, identity, identity],
                                pipeline_depth=4,
                                inputs=[1.0],
                                optimizer_function=optimizer_function,
                                outfeed_loss=True)

  @test_util.deprecated_graph_mode_only
  def testOutfeedLoss(self):

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

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed12")

      def my_net(x):
        return pipelining_ops.pipeline([stage1, identity, identity, identity],
                                       pipeline_depth=8,
                                       inputs=[x],
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=optimizer_function,
                                       outfeed_loss=True)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[0.0])

      cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
      cfg = utils.set_ipu_model_options(cfg,
                                        compile_ipu_code=True,
                                        tiles_per_ipu=128)
      cfg = utils.auto_select_ipus(cfg, 4)
      utils.configure_ipu_system(cfg)
      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      self.assertAllEqual(np.ones(8), sess.run(outfed))

  @test_util.deprecated_graph_mode_only
  def testOutfeedDict(self):

    with tu.ipu_session() as sess:

      def identity(x):
        return x

      def dictstage(x):
        return {"x": x}

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed13")

      def my_net(x):
        return pipelining_ops.pipeline(
            [identity, identity, identity, dictstage],
            pipeline_depth=8,
            inputs=[x],
            outfeed_queue=outfeed_queue)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0])

      cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
      cfg = utils.set_ipu_model_options(cfg,
                                        compile_ipu_code=True,
                                        tiles_per_ipu=128)
      cfg = utils.auto_select_ipus(cfg, 4)
      utils.configure_ipu_system(cfg)
      utils.move_variable_initialization_to_cpu()

      outfed = outfeed_queue.dequeue()

      sess.run(variables.global_variables_initializer())
      sess.run(pipeline)
      got = sess.run(outfed)
      self.assertIsInstance(got, dict)
      self.assertAllEqual(np.ones(8), got["x"])

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

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed14")

      def my_net(x):
        return pipelining_ops.pipeline([stage1, stage2],
                                       pipeline_depth=4,
                                       inputs=[x],
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=optimizer_function)

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

      outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed15")

      def my_net(x):
        return pipelining_ops.pipeline([stage1, identity, identity, identity],
                                       pipeline_depth=8,
                                       inputs=[x],
                                       outfeed_queue=outfeed_queue,
                                       optimizer_function=optimizer_function,
                                       outfeed_loss=True)

      with ops.device("/device:IPU:0"):
        pipeline = ipu_compiler.compile(my_net, inputs=[1.0])

      cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
      cfg = utils.set_ipu_model_options(cfg,
                                        compile_ipu_code=True,
                                        tiles_per_ipu=128)
      cfg = utils.auto_select_ipus(cfg, 4)
      utils.configure_ipu_system(cfg)
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
    infeed_queue = ipu_infeed_queue.IPUInfeedQueue(dataset, "__feed16")
    outfeed_queue = ipu_outfeed_queue.IPUOutfeedQueue("__feed16")

    def stage1(x):
      return x

    def stage2(x):
      return x

    def stage3(x):
      p = x > 2.0
      return control_flow_ops.cond(p, lambda: constant_op.constant(1.0),
                                   lambda: constant_op.constant(2.0))

    def my_net():
      return pipelining_ops.pipeline([stage1, stage2, stage3],
                                     6,
                                     inputs=[],
                                     infeed_queue=infeed_queue,
                                     outfeed_queue=outfeed_queue)

    with tu.ipu_session() as sess:
      with ops.device("/device:IPU:0"):
        r = ipu_compiler.compile(my_net)

      cfg = utils.create_ipu_config(profiling=True, profile_execution=True)
      cfg = utils.set_ipu_model_options(cfg,
                                        compile_ipu_code=True,
                                        tiles_per_ipu=128)
      cfg = utils.auto_select_ipus(cfg, 4)
      utils.configure_ipu_system(cfg)
      utils.move_variable_initialization_to_cpu()

      outfeed_op = outfeed_queue.dequeue()
      sess.run(infeed_queue.initializer)
      sess.run(r)
      output = sess.run(outfeed_op)
      self.assertAllClose(output, [[2.0, 2.0, 2.0, 1.0, 1.0, 1.0]])

  @test_util.deprecated_graph_mode_only
  def testPipelineWithCustomGradientFunction(self):
    def dataset_fn():
      dataset = tu.create_single_increasing_dataset(10, shape=[4])
      dataset = dataset.batch(batch_size=4, drop_remainder=True)

      def dataset_parser(value):
        label = math_ops.reduce_mean(value, axis=[1])
        return value, math_ops.cast(label / 10, np.int32)

      return dataset.map(dataset_parser)

    pipeline_depth = 24
    repeat_count = 2
    optimizer = gradient_descent.GradientDescentOptimizer(0.01)

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
          nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=label))
      return loss

    def inputs_fn():
      with ops.device('cpu'):
        return []

    pipelining_test_util.PipelineTester.compare_pipeline_to_cpu(
        [stage1, stage2, stage3],
        inputs_fn, [],
        repeat_count,
        pipeline_depth,
        dataset_fn,
        optimizer,
        self,
        14415,
        schedule=pipelining_ops.PipelineSchedule.Grouped)


if __name__ == "__main__":
  googletest.main()
